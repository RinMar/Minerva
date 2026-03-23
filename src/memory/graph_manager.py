import json
from collections import deque
from src.memory.schema import EntityNode, GraphEdge, EmbeddingIndex
from src.memory.db import get_session
from src.memory.hash_utils import fact_id


def _upsert_entity(session, name: str, topic: str, text: str, tags: list, user_name: str, emb_model):
    """Get or create entity node, update summary, and maintain embedding."""
    eid = fact_id(name.lower())
    entity = session.query(EntityNode).filter_by(id=eid, user_name=user_name).first()

    if entity:
        # Append text if it's new information
        if text not in entity.text:
            entity.text = entity.text + "\n" + text

            # Update embedding since text changed
            emb = emb_model.encode(entity.text).tolist()
            emb_record = session.query(EmbeddingIndex).filter_by(
                user_name=user_name, collection="entity", source_id=eid).first()
            if emb_record:
                emb_record.text_content = entity.text
                emb_record.embedding_json = json.dumps(emb)
    else:
        # Create new entity
        entity = EntityNode(
            id=eid,
            user_name=user_name,
            name=name,
            topic=topic,
            text=text,
            tags_json=json.dumps(tags)
        )
        session.add(entity)

        # Add embedding
        emb = emb_model.encode(entity.text).tolist()
        new_emb = EmbeddingIndex(
            user_name=user_name,
            collection="entity",
            source_id=eid,
            text_content=entity.text,
            embedding_json=json.dumps(emb)
        )
        session.add(new_emb)

    return eid


def add_triplets(triplets: list, topic: str, tags: list, fact_text: str, user_name: str, emb_model):
    """
    Ingest extracted triplets into Graph EntityNodes and GraphEdges.
    Handles upserting entities and creating edges + edge embeddings.
    """
    if not triplets:
        # If no triplets extracted, we create a single standalone node based on the topic
        with get_session() as session:
            _upsert_entity(session, topic, topic, fact_text, tags, user_name, emb_model)
            session.commit()
        return 0

    added_edges = 0
    with get_session() as session:
        for triplet in triplets:
            head = triplet.get("head")
            relation = triplet.get("type")
            tail = triplet.get("tail")

            if not head or not tail or not relation:
                continue

            # Upsert Nodes
            head_id = _upsert_entity(session, head, topic, fact_text, tags, user_name, emb_model)
            tail_id = _upsert_entity(session, tail, topic, fact_text, tags, user_name, emb_model)

            # Deduplicate edge
            existing_edge = session.query(GraphEdge).filter_by(
                user_name=user_name,
                source_id=head_id,
                target_id=tail_id,
                relation=relation
            ).first()

            if not existing_edge:
                edge = GraphEdge(
                    user_name=user_name,
                    source_id=head_id,
                    target_id=tail_id,
                    relation=relation
                )
                session.add(edge)
                session.flush()  # assign edge.id

                # Add edge embedding
                triplet_text = f"{head} {relation} {tail}"
                emb = emb_model.encode(triplet_text).tolist()
                new_emb = EmbeddingIndex(
                    user_name=user_name,
                    collection="edge",
                    source_id=str(edge.id),
                    text_content=triplet_text,
                    embedding_json=json.dumps(emb)
                )
                session.add(new_emb)
                added_edges += 1

        session.commit()
    return added_edges


def expand_nodes(seed_ids: list, user_name: str, depth: int = 1, max_nodes: int = 20) -> list:
    """
    BFS expansion from seed entity ids through the GraphEdge graph.
    Returns a list of connected entity ids (including seeds).
    Supports bidirectional traversal.
    """
    if not seed_ids:
        return []

    visited = set(seed_ids)
    queue = deque()
    for sid in seed_ids:
        queue.append((sid, 0))

    with get_session() as session:
        while queue and len(visited) < max_nodes:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            # Outgoing edges
            outgoing = session.query(GraphEdge.target_id).filter_by(
                user_name=user_name,
                source_id=current_id
            ).all()

            # Incoming edges (reverse traversal)
            incoming = session.query(GraphEdge.source_id).filter_by(
                user_name=user_name,
                target_id=current_id
            ).all()

            neighbors = [row[0] for row in outgoing] + [row[0] for row in incoming]

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

    return list(visited)
