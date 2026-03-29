"""
Tool execution utilities for the conversational RAG interface.
Handles the specific JSON payload unmarshalling and memory dispatch logic
for `retrieve` and `manage_memory` tools.
"""
import src.memory.store as memory_store
import src.memory.retrieve as memory_retrieve

def execute_retrieve(tool_args: dict, user_id: int, emb_model, cross_encoder) -> str:
    query = tool_args.get("query", "")
    if not query:
        return "Error: No query provided."

    search_emb = emb_model.encode(query).tolist()
    context = memory_retrieve.retrieve_context(
        query_emb=search_emb,
        user_id=user_id,
        cross_encoder=cross_encoder,
        query_text=query,
    )

    if context:
        return f"Retrieved Context:\n{context}"
    return "No relevant context found."


def execute_manage_memory(tool_args: dict, user_id: int, emb_model) -> str:
    count = 0
    count += _process_stores(tool_args.get("store", []), user_id, emb_model)
    count += _process_add_entities(tool_args.get("add_entity", []), user_id, emb_model)
    count += _process_add_edges(tool_args.get("add_edge", []), user_id, emb_model)
    count += _process_delete_edges(tool_args.get("delete_edge", []), user_id)
    count += _process_update_edges(tool_args.get("update_edge", []), user_id, emb_model)
    count += _process_delete_entities(tool_args.get("delete_entity", []), user_id)
    count += _process_update_entities(tool_args.get("update_entity", []), user_id, emb_model)
    return f"Successfully executed {count} memory operations synchronously."


def _process_stores(stores: list, user_id: int, emb_model) -> int:
    count = 0
    for fact_obj in stores:
        topic = fact_obj.get("topic", "general")
        fact = fact_obj.get("fact", "")
        tags = fact_obj.get("tags", ["tool_stored"])
        triplets = fact_obj.get("triplets", [])
        if triplets or fact:
            memory_store.store_triplets(triplets, topic, tags, fact, user_id, emb_model)
            count += 1
    return count


def _process_add_entities(entities: list, user_id: int, emb_model) -> int:
    count = 0
    for ent in entities:
        name = ent.get("name")
        text = ent.get("text", "")
        topic = ent.get("topic", "general")
        tags = ent.get("tags", ["tool_stored"])
        if name:
            memory_store.add_entity(name, topic, text, tags, user_id, emb_model)
            count += 1
    return count


def _process_add_edges(edges: list, user_id: int, emb_model) -> int:
    count = 0
    for edge in edges:
        s, t, r = edge.get("source"), edge.get("target"), edge.get("relation")
        if s and t and r:
            memory_store.add_edge(s, t, r, user_id, emb_model)
            count += 1
    return count


def _process_delete_edges(edges: list, user_id: int) -> int:
    count = 0
    for edge_obj in edges:
        s, t, r = edge_obj.get("source"), edge_obj.get("target"), edge_obj.get("relation")
        if s and t and r:
            memory_store.delete_edge(s, t, r, user_id)
            count += 1
    return count


def _process_update_edges(edges: list, user_id: int, emb_model) -> int:
    count = 0
    for edge_obj in edges:
        s, t = edge_obj.get("source"), edge_obj.get("target")
        old_r, new_r = edge_obj.get("old_relation"), edge_obj.get("new_relation")
        if s and t and old_r and new_r:
            memory_store.update_edge(s, t, old_r, new_r, emb_model, user_id)
            count += 1
    return count


def _process_delete_entities(entities: list, user_id: int) -> int:
    count = 0
    for ent_obj in entities:
        name = ent_obj.get("name")
        if name:
            memory_store.delete_entity(name, user_id)
            count += 1
    return count


def _process_update_entities(entities: list, user_id: int, emb_model) -> int:
    count = 0
    for ent_obj in entities:
        old_name, new_name = ent_obj.get("old_name"), ent_obj.get("new_name")
        new_text = ent_obj.get("new_text")
        if old_name and (new_name or new_text):
            memory_store.update_entity(old_name, new_name, new_text, emb_model, user_id)
            count += 1
    return count
