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


def _find_closest_entity(session, text: str, emb_model, user_name: str, threshold: float = 0.85) -> str:
    """Find the most semantically similar EntityNode in the DB for a given text."""
    query_emb = emb_model.encode(text).tolist()
    all_entities = session.query(EmbeddingIndex.source_id, EmbeddingIndex.embedding_json).filter_by(
        user_name=user_name, collection="entity"
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


def delete_fact(fact_text: str, emb_model, user_name: str, memory_base_dir="local_store"):
    """
    Delete a fact from the database entirely.
    Removes the most similar edge and removes the exact or highly similar text from EntityNodes.
    """
    import json
    with get_session() as session:
        query_emb = emb_model.encode(fact_text).tolist()
        deleted_something = False

        # 1. Find and delete the most similar edge (threshold 0.85)
        all_edges = session.query(EmbeddingIndex.source_id, EmbeddingIndex.embedding_json).filter_by(
            user_name=user_name, collection="edge"
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
        all_entities = session.query(EntityNode).filter_by(user_name=user_name).all()
        
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


def retrieve_context(query_emb, user_name: str, cross_encoder=None,
                     query_text: str = "", k: int = 10, top_n: int = 5,
                     memory_base_dir="local_store") -> str:
    with get_session() as session:
        # Load all embeddings (entities and edges) for the user
        all_embs = session.query(EmbeddingIndex).filter_by(
            user_name=user_name
        ).all()

        if not all_embs:
            return ""

        # Step 1: Embedding search
        scored = []
        for row in all_embs:
            emb = json.loads(row.embedding_json)
            score = cosine_similarity(query_emb, emb)
            scored.append((row, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k_rows = [row for row, _ in scored[:k]]

        if not top_k_rows:
            return ""

        # Step 2: Extract seed Entity IDs from the top-k result
        seed_entity_ids = []
        for row in top_k_rows:
            if row.collection == "entity":
                seed_entity_ids.append(row.source_id)
            elif row.collection == "edge":
                # Find the target edge and get its source/target entities
                edge = session.query(GraphEdge).filter_by(id=int(row.source_id)).first()
                if edge:
                    seed_entity_ids.extend([edge.source_id, edge.target_id])

        seed_entity_ids = list(set(seed_entity_ids))

        if not seed_entity_ids:
            return ""

        # Step 3: Fact-to-Fact Graph expansion
        expanded_ids = graph_manager.expand_nodes(
            seed_entity_ids, user_name, depth=1, max_nodes=20
        )

        final_ids = list(set(seed_entity_ids).union(expanded_ids))

        # Step 4: Load texts for the expanded candidates
        entity_rows = session.query(EntityNode.id, EntityNode.text).filter(
            EntityNode.user_name == user_name,
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

        # Step 5: Format
        if not candidates:
            return ""

        lines = ["Relevant known facts:\n"]
        for eid, text in candidates:
            lines.append(f"- {text}")
        return "\n".join(lines)
