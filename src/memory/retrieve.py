"""
Memory Query Operations — Non-destructive graph traversals and semantic search.
"""

import json
from collections import deque
from src.memory.db import EntityNode, GraphEdge, EmbeddingIndex, get_session
from src.utils import cosine_similarity


def expand_nodes(seed_ids: list, user_id: int, depth: int = 1, max_nodes: int = 20) -> list:
    if not seed_ids:
        return []
    visited = set(seed_ids)
    queue = deque([(sid, 0) for sid in seed_ids])

    with get_session() as session:
        while queue and len(visited) < max_nodes:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            outgoing = session.query(GraphEdge.target_id).filter_by(user_id=user_id, source_id=current_id).all()
            incoming = session.query(GraphEdge.source_id).filter_by(user_id=user_id, target_id=current_id).all()
            neighbors = [r[0] for r in outgoing] + [r[0] for r in incoming]

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))
    return list(visited)


def _rerank(cross_encoder, query_text: str, candidates: list, top_n: int) -> list:
    if not candidates:
        return []
    pairs = [(query_text, text) for _, text in candidates]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored][:top_n]


def _score_embeddings(all_embs, query_emb, k: int) -> list:
    scored = []
    for row in all_embs:
        emb = json.loads(row.embedding_json)
        score = cosine_similarity(query_emb, emb)
        scored.append((row, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:k]]


def _extract_seed_entity_ids(session, top_k_rows) -> list:
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


def retrieve_context(query_emb, user_id: int, cross_encoder=None, query_text: str = "",
                     k: int = 10, top_n: int = 5) -> str:
    with get_session() as session:
        all_embs = session.query(EmbeddingIndex).filter_by(user_id=user_id).all()
        if not all_embs:
            return ""

        top_k_rows = _score_embeddings(all_embs, query_emb, k)
        if not top_k_rows:
            return ""

        seed_entity_ids = _extract_seed_entity_ids(session, top_k_rows)
        if not seed_entity_ids:
            return ""

        expanded_ids = expand_nodes(seed_entity_ids, user_id, depth=1, max_nodes=20)
        final_ids = list(set(seed_entity_ids).union(expanded_ids))

        entity_rows = session.query(EntityNode.id, EntityNode.name, EntityNode.text).filter(
            EntityNode.user_id == user_id, EntityNode.id.in_(final_ids)).all()

        candidates = [(r.id, r.text) for r in entity_rows]
        if cross_encoder and query_text:
            candidates = _rerank(cross_encoder, query_text, candidates, top_n)
        else:
            candidates = candidates[:top_n]

        name_map = {r.id: r.name for r in entity_rows}
        edges = session.query(GraphEdge).filter(
            GraphEdge.user_id == user_id,
            GraphEdge.source_id.in_(final_ids),
            GraphEdge.target_id.in_(final_ids)
        ).all()

        return _format_context_results(candidates, edges, name_map)
