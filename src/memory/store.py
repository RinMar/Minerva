"""
Memory Storage Operations — Graph DB mutation methods.
"""

import json
import re
from src.memory.db import EntityNode, GraphEdge, EmbeddingIndex, get_session
from src.utils import fact_id


def _upsert_entity(session, name: str, topic: str, text: str, tags: list, user_id: int, emb_model):
    """Get or create entity node, update summary, and maintain embedding."""
    eid = fact_id(name.lower(), user_id)
    entity = session.query(EntityNode).filter_by(id=eid, user_id=user_id).first()

    if entity:
        if text and text not in entity.text:
            entity.text = entity.text + "\n" + text
            emb = emb_model.encode(entity.text).tolist()
            emb_record = session.query(EmbeddingIndex).filter_by(
                user_id=user_id, collection="entity", source_id=eid).first()
            if emb_record:
                emb_record.text_content = entity.text
                emb_record.embedding_json = json.dumps(emb)
    else:
        entity = EntityNode(
            id=eid,
            user_id=user_id,
            name=name,
            topic=topic,
            text=text or name,
            tags_json=json.dumps(tags)
        )
        session.add(entity)
        emb = emb_model.encode(entity.text).tolist()
        new_emb = EmbeddingIndex(
            user_id=user_id,
            collection="entity",
            source_id=eid,
            text_content=entity.text,
            embedding_json=json.dumps(emb)
        )
        session.add(new_emb)
        session.flush()

    return eid


def add_entity(name: str, topic: str, text: str, tags: list, user_id: int, emb_model):
    """Explicitly add or append text to an entity node. Returns True on success."""
    if not name:
        return False
    with get_session() as session:
        _upsert_entity(session, name, topic, text, tags, user_id, emb_model)
        session.commit()
    return True


def store_triplets(triplets: list, topic: str, tags: list, fact_text: str, user_id: int, emb_model):
    """Ingest extracted triplets into Graph EntityNodes and GraphEdges. Returns count of added edges."""
    if not triplets:
        with get_session() as session:
            _upsert_entity(session, topic, topic, fact_text, tags, user_id, emb_model)
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

            head_id = _upsert_entity(session, head, topic, fact_text, tags, user_id, emb_model)
            tail_id = _upsert_entity(session, tail, topic, fact_text, tags, user_id, emb_model)

            existing_edge = session.query(GraphEdge).filter_by(
                user_id=user_id, source_id=head_id, target_id=tail_id, relation=relation
            ).first()

            if not existing_edge:
                edge = GraphEdge(
                    user_id=user_id,
                    source_id=head_id,
                    target_id=tail_id,
                    source_name=head,
                    target_name=tail,
                    relation=relation
                )
                session.add(edge)
                session.flush()

                triplet_text = f"{head} {relation} {tail}"
                emb = emb_model.encode(triplet_text).tolist()
                new_emb = EmbeddingIndex(
                    user_id=user_id,
                    collection="edge",
                    source_id=str(edge.id),
                    text_content=triplet_text,
                    embedding_json=json.dumps(emb)
                )
                session.add(new_emb)
                added_edges += 1

        session.commit()
    return added_edges


def add_edge(source: str, target: str, relation: str, user_id: int, emb_model):
    """Explicitly add an edge between two entities. Returns count of added edges."""
    if not source or not target or not relation:
        return 0
    return store_triplets([{"head": source, "type": relation, "tail": target}], "general", [], "", user_id, emb_model)


def delete_edge(source_name: str, target_name: str, relation: str, user_id: int):
    """Delete edges matching the criteria. Returns count of deleted edges."""
    if not source_name or not target_name or not relation:
        return 0
    deleted_count = 0
    with get_session() as session:
        edges = session.query(GraphEdge).filter(
            GraphEdge.user_id == user_id, GraphEdge.relation == relation).all()
        for edge in edges:
            if edge.source_name.lower() == source_name.lower() and edge.target_name.lower() == target_name.lower():
                session.delete(edge)
                session.query(EmbeddingIndex).filter_by(source_id=str(edge.id), collection="edge").delete()
                deleted_count += 1
        session.commit()
    return deleted_count


def update_edge(source_name: str, target_name: str, old_relation: str, new_relation: str, emb_model, user_id: int):
    """Update edges matching the criteria. Returns count of updated edges."""
    if not source_name or not target_name or not old_relation or not new_relation:
        return 0
    updated_count = 0
    with get_session() as session:
        edges = session.query(GraphEdge).filter(
            GraphEdge.user_id == user_id, GraphEdge.relation == old_relation).all()
        for edge in edges:
            if edge.source_name.lower() == source_name.lower() and edge.target_name.lower() == target_name.lower():
                edge.relation = new_relation
                triplet_text = f"{edge.source_name} {new_relation} {edge.target_name}"
                emb = emb_model.encode(triplet_text).tolist()
                session.query(EmbeddingIndex).filter_by(source_id=str(edge.id), collection="edge").update({
                    "text_content": triplet_text, "embedding_json": json.dumps(emb)
                })
                updated_count += 1
        session.commit()
    return updated_count


def delete_entity(name: str, user_id: int):
    """Delete entities and their associated edges. Returns count of deleted entities."""
    if not name:
        return 0
    deleted_count = 0
    with get_session() as session:
        entities = session.query(EntityNode).filter_by(user_id=user_id).all()
        for e in entities:
            if e.name.lower() == name.lower():
                session.delete(e)
                session.query(EmbeddingIndex).filter_by(source_id=e.id, collection="entity").delete()
                attached_edges = session.query(GraphEdge).filter(
                    (GraphEdge.source_id == e.id) | (GraphEdge.target_id == e.id)).all()
                for edge in attached_edges:
                    session.delete(edge)
                    session.query(EmbeddingIndex).filter_by(source_id=str(edge.id), collection="edge").delete()
                deleted_count += 1
        session.commit()
    return deleted_count


def update_entity(old_name: str, new_name: str, new_text: str, emb_model, user_id: int):
    """Update entity properties. Returns count of updated entities."""
    if not old_name:
        return 0
    updated_count = 0
    with get_session() as session:
        entities = session.query(EntityNode).filter_by(user_id=user_id).all()
        for e in entities:
            if e.name.lower() == old_name.lower():
                if new_name:
                    e.name = new_name
                if new_text:
                    e.text = new_text
                elif new_name:
                    e.text = re.sub(r'(?i)\b' + re.escape(old_name) + r'\b', new_name, e.text)
                    if not e.text.strip():
                        e.text = new_name
                emb = emb_model.encode(e.text).tolist()
                session.query(EmbeddingIndex).filter_by(source_id=e.id, collection="entity").update({
                    "text_content": e.text, "embedding_json": json.dumps(emb)
                })
                if new_name:
                    attached_edges = session.query(GraphEdge).filter(
                        (GraphEdge.source_id == e.id) | (GraphEdge.target_id == e.id)).all()
                    for edge in attached_edges:
                        if edge.source_id == e.id:
                            edge.source_name = new_name
                        elif edge.target_id == e.id:
                            edge.target_name = new_name
                        triplet_text = f"{edge.source_name} {edge.relation} {edge.target_name}"
                        edge_emb = emb_model.encode(triplet_text).tolist()
                        session.query(EmbeddingIndex).filter_by(source_id=str(edge.id), collection="edge").update({
                            "text_content": triplet_text, "embedding_json": json.dumps(edge_emb)
                        })
                updated_count += 1
        session.commit()
    return updated_count
