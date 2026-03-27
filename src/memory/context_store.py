"""
Context store — storage and retrieval via SQLite + SQLAlchemy.

STORAGE:
  - add_fact: write to SQLite (dedup by fact_id)
  - update_fact: delete old, insert new
  - delete_fact: natively remove fact (cascades to graph edges)

RETRIEVAL:
  - retrieve_context:
      fetch all user embeddings → compute similarity → rerank seeds → graph expand → rerank final → format
"""

import json
from src.utils import cosine_similarity
from src.memory import graph_manager
from src.memory.db import get_session
from src.memory.db import EntityNode, EmbeddingIndex, GraphEdge


def _find_closest_entity(session, text: str, emb_model, user_id: int, threshold: float = 0.85) -> str:
    """Find the most semantically similar EntityNode in the DB for a given text."""
    query_emb = emb_model.encode(text).tolist()
    all_entities = session.query(EmbeddingIndex.source_id, EmbeddingIndex.embedding_json).filter_by(
        user_id=user_id, collection="entity"
    ).all()

    best_id = None
    best_score = -1
    for eid, emb_json in all_entities:
        emb = json.loads(emb_json)
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_id = eid

    if best_score >= threshold:
        return best_id
    return None


def delete_fact(fact_text: str, emb_model, user_id: int):
    """
    Delete a fact from the database entirely.
    Removes the most similar edge and removes the exact or highly similar text from EntityNodes.
    """
    with get_session() as session:
        query_emb = emb_model.encode(fact_text).tolist()
        deleted_something = False

        # 1. Find and delete the most similar edge (threshold 0.85)
        all_edges = session.query(EmbeddingIndex.source_id, EmbeddingIndex.embedding_json).filter_by(
            user_id=user_id, collection="edge"
        ).all()

        edges_to_delete = []
        for eid, emb_json in all_edges:
            emb = json.loads(emb_json)
            score = cosine_similarity(query_emb, emb)
            if score >= 0.85:
                edges_to_delete.append(eid)

        for eid in edges_to_delete:
            session.query(GraphEdge).filter_by(id=int(eid)).delete()
            session.query(EmbeddingIndex).filter_by(source_id=eid, collection="edge").delete()
            deleted_something = True

        # 2. Remove fact text from any EntityNode that has it
        all_entities = session.query(EntityNode).filter_by(user_id=user_id).all()

        for e in all_entities:
            lines = e.text.split('\n')
            new_lines = []
            changed = False
            for line in lines:
                if not line.strip():
                    continue
                # Substring match
                if fact_text.lower() in line.lower() or line.lower() in fact_text.lower():
                    changed = True
                    continue

                # Semantic match
                line_emb = emb_model.encode(line).tolist()
                score = cosine_similarity(query_emb, line_emb)
                if score >= 0.85:
                    changed = True
                else:
                    new_lines.append(line)

            if changed:
                deleted_something = True
                if not new_lines:
                    session.delete(e)
                    session.query(EmbeddingIndex).filter_by(source_id=e.id, collection="entity").delete()
                    # Delete attached edges manually since no CASCADE enforced
                    attached_edges = session.query(GraphEdge).filter(
                        (GraphEdge.source_id == e.id) | (GraphEdge.target_id == e.id)
                    ).all()
                    for edge in attached_edges:
                        session.delete(edge)
                        session.query(EmbeddingIndex).filter_by(source_id=str(edge.id), collection="edge").delete()
                else:
                    e.text = '\n'.join(new_lines)
                    emb = emb_model.encode(e.text).tolist()
                    session.query(EmbeddingIndex).filter_by(source_id=e.id, collection="entity").update({
                        "text_content": e.text,
                        "embedding_json": json.dumps(emb)
                    })

        session.commit()
        return deleted_something


def _rerank(cross_encoder, query_text: str, candidates: list,
            top_n: int) -> list:
    """Score candidates with the cross-encoder and sort."""
    if not candidates:
        return []
    pairs = [(query_text, text) for _, text in candidates]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, score in scored][:top_n]


def _score_embeddings(all_embs, query_emb, k: int) -> list:
    """Compute cosine similarity and return top-k rows."""
    scored = []
    for row in all_embs:
        emb = json.loads(row.embedding_json)
        score = cosine_similarity(query_emb, emb)
        scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:k]]


def _extract_seed_entity_ids(session, top_k_rows) -> list:
    """Extract entity IDs from a mix of entity and edge rows."""
    seed_entity_ids = []
    for row in top_k_rows:
        if row.collection == "entity":
            seed_entity_ids.append(row.source_id)
        elif row.collection == "edge":
            edge = session.query(GraphEdge).filter_by(id=int(row.source_id)).first()
            if edge:
                seed_entity_ids.extend([edge.source_id, edge.target_id])
    return list(set(seed_entity_ids))


def _format_context_results(candidates, edges, name_map) -> str:
    """Format candidates and edges into a readable context string."""
    if not candidates and not edges:
        return ""

    lines = []
    if candidates:
        lines.append("Relevant entities:")
        for eid, text in candidates:
            name = name_map.get(eid, eid)
            flat = text.replace("\n", ". ").strip()
            lines.append(f"[{name}]: {flat}")

    if edges:
        lines.append("\nRelations:")
        for e in edges:
            lines.append(f"- {e.source_name} -- {e.relation} --> {e.target_name}")

    return "\n".join(lines)


def retrieve_context(query_emb, user_id: int, cross_encoder=None,
                     query_text: str = "", k: int = 10, top_n: int = 5,
                     ) -> str:
    with get_session() as session:
        # Load all embeddings (entities and edges) for the user
        all_embs = session.query(EmbeddingIndex).filter_by(
            user_id=user_id
        ).all()

        if not all_embs:
            return ""

        # Step 1: Embedding search
        top_k_rows = _score_embeddings(all_embs, query_emb, k)
        if not top_k_rows:
            return ""

        # Step 2: Extract seed Entity IDs from the top-k result
        seed_entity_ids = _extract_seed_entity_ids(session, top_k_rows)
        if not seed_entity_ids:
            return ""

        # Step 3: Fact-to-Fact Graph expansion
        expanded_ids = graph_manager.expand_nodes(
            seed_entity_ids, user_id, depth=1, max_nodes=20
        )

        final_ids = list(set(seed_entity_ids).union(expanded_ids))

        # Step 4a: Load and rerank top entity nodes
        entity_rows = session.query(
            EntityNode.id, EntityNode.name, EntityNode.text
        ).filter(
            EntityNode.user_id == user_id,
            EntityNode.id.in_(final_ids)
        ).all()

        candidates = [(r.id, r.text) for r in entity_rows]

        if cross_encoder and query_text:
            candidates = _rerank(
                cross_encoder, query_text, candidates,
                top_n=top_n
            )
        else:
            candidates = candidates[:top_n]

        # Build a name lookup for formatting
        name_map = {r.id: r.name for r in entity_rows}

        # Step 4b: Load graph edges between all expanded entities
        edges = session.query(GraphEdge).filter(
            GraphEdge.user_id == user_id,
            GraphEdge.source_id.in_(final_ids),
            GraphEdge.target_id.in_(final_ids)
        ).all()

        # Step 5: Format as two sections
        return _format_context_results(candidates, edges, name_map)
