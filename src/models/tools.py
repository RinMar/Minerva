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
    reports = []
    reports.extend(_process_stores(tool_args.get("store", []), user_id, emb_model))
    reports.extend(_process_add_entities(tool_args.get("add_entity", []), user_id, emb_model))
    reports.extend(_process_add_edges(tool_args.get("add_edge", []), user_id, emb_model))
    reports.extend(_process_delete_edges(tool_args.get("delete_edge", []), user_id))
    reports.extend(_process_update_edges(tool_args.get("update_edge", []), user_id, emb_model))
    reports.extend(_process_delete_entities(tool_args.get("delete_entity", []), user_id))
    reports.extend(_process_update_entities(tool_args.get("update_entity", []), user_id, emb_model))

    if not reports:
        return "No memory operations were requested or executed."

    success_count = sum(1 for r in reports if not r.startswith("Warning") and not r.startswith("Error"))
    summary = f"Executed {success_count} / {len(reports)} operations successfully.\n\nDetailed Report:\n"
    summary += "\n".join([f"- {r}" for r in reports])
    return summary


def _process_stores(stores: list, user_id: int, emb_model) -> list:
    reports = []
    for fact_obj in stores:
        topic = fact_obj.get("topic", "general")
        fact = fact_obj.get("fact", "")
        tags = fact_obj.get("tags", ["tool_stored"])
        triplets = fact_obj.get("triplets", [])
        if triplets or fact:
            added = memory_store.store_triplets(triplets, topic, tags, fact, user_id, emb_model)
            if added > 0 or not triplets:
                reports.append(f"Stored fact about '{topic}': {fact}")
            else:
                reports.append(f"Warning: No new triplets added for topic '{topic}'.")
    return reports


def _process_add_entities(entities: list, user_id: int, emb_model) -> list:
    reports = []
    for ent in entities:
        name = ent.get("name")
        text = ent.get("text", "")
        topic = ent.get("topic", "general")
        tags = ent.get("tags", ["tool_stored"])
        if name:
            success = memory_store.add_entity(name, topic, text, tags, user_id, emb_model)
            if success:
                reports.append(f"Added/Updated entity: {name}")
            else:
                reports.append(f"Error: Failed to add entity: {name}")
    return reports


def _process_add_edges(edges: list, user_id: int, emb_model) -> list:
    reports = []
    for edge in edges:
        s, t, r = edge.get("source"), edge.get("target"), edge.get("relation")
        if s and t and r:
            added = memory_store.add_edge(s, t, r, user_id, emb_model)
            if added > 0:
                reports.append(f"Added edge: {s} -- {r} --> {t}")
            else:
                reports.append(f"Warning: Edge already exists: {s} -- {r} --> {t}")
    return reports


def _process_delete_edges(edges: list, user_id: int) -> list:
    reports = []
    for edge_obj in edges:
        s, t, r = edge_obj.get("source"), edge_obj.get("target"), edge_obj.get("relation")
        if s and t and r:
            count = memory_store.delete_edge(s, t, r, user_id)
            if count > 0:
                reports.append(f"Deleted {count} edge(s): {s} -- {r} --> {t}")
            else:
                reports.append(f"Warning: Could not find edge to delete: {s} -- {r} --> {t}")
    return reports


def _process_update_edges(edges: list, user_id: int, emb_model) -> list:
    reports = []
    for edge_obj in edges:
        s, t = edge_obj.get("source"), edge_obj.get("target")
        old_r, new_r = edge_obj.get("old_relation"), edge_obj.get("new_relation")
        if s and t and old_r and new_r:
            count = memory_store.update_edge(s, t, old_r, new_r, emb_model, user_id)
            if count > 0:
                reports.append(f"Updated {count} edge(s) from '{old_r}' to '{new_r}': {s} -> {t}")
            else:
                reports.append(f"Warning: Could not find edge to update: {s} -- {old_r} --> {t}")
    return reports


def _process_delete_entities(entities: list, user_id: int) -> list:
    reports = []
    for ent_obj in entities:
        name = ent_obj.get("name")
        if name:
            count = memory_store.delete_entity(name, user_id)
            if count > 0:
                reports.append(f"Deleted entity: {name}")
            else:
                reports.append(f"Warning: Could not find entity to delete: {name}")
    return reports


def _process_update_entities(entities: list, user_id: int, emb_model) -> list:
    reports = []
    for ent_obj in entities:
        old_name, new_name = ent_obj.get("old_name"), ent_obj.get("new_name")
        new_text = ent_obj.get("new_text")
        if old_name and (new_name or new_text):
            count = memory_store.update_entity(old_name, new_name, new_text, emb_model, user_id)
            if count > 0:
                reports.append(f"Updated entity: {old_name}" + (f" -> {new_name}" if new_name else ""))
            else:
                reports.append(f"Warning: Could not find entity to update: {old_name}")
    return reports
