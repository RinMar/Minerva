"""
Memory orchestrator — background storage pipeline.
Used to provide non-blocking asynchronous processing of facts and memory graph updates
so the main chat interface remains responsive.
"""

import threading


class MemoryOrchestrator:
    def __init__(self, llm, emb_model, user_name: str, cross_encoder=None):
        self.llm = llm
        self.emb_model = emb_model
        self.user_name = user_name
        self.cross_encoder = cross_encoder
        self._pending_facts = []

    def trigger_store(self, topic: str, fact: str = "", old_fact: str = "", new_fact: str = "", action: str = "store", triplets: list = None):
        """Queue a memory operation to be processed."""
        item = {"action": action, "topic": topic}
        if action == "store":
            item["text"] = fact
            item["triplets"] = triplets
        elif action == "update":
            item["old_fact"] = old_fact
            item["new_fact"] = new_fact
        elif action == "delete":
            item["text"] = fact
        self._pending_facts.append(item)

    def trigger_update_fact(self, topic: str, old_fact: str, new_fact: str, triplets: list = None):
        self._pending_facts.append({"action": "update", "topic": topic, "old_fact": old_fact, "new_fact": new_fact, "triplets": triplets})

    def trigger_delete_fact(self, fact: str):
        self._pending_facts.append({"action": "delete", "text": fact})

    def process_pending_stores(self):
        """Run the storage pipeline in a background thread."""
        if not self._pending_facts:
            return

        facts_to_process = list(self._pending_facts)
        self._pending_facts.clear()

        threading.Thread(
            target=self._run_store_pipeline,
            args=(facts_to_process,),
            daemon=True
        ).start()

    def _run_store_pipeline(self, facts: list):
        """Process store/update/delete operations and build FactEdge graph."""
        for fact in facts:
            action = fact.get("action", "store")
            topic = fact.get("topic", "general")

            try:
                if action == "store":
                    self._process_store(fact, topic)
                elif action == "update":
                    self._process_update(fact, topic)
                elif action == "delete":
                    self._process_delete(fact)
            except Exception as e:
                print(f"[Orchestrator] Store pipeline error: {e}")

    def _process_store(self, fact: dict, topic: str):
        from src.memory import graph_manager
        from src.memory.triplet_extraction import extract_triplets

        text = fact.get("fact", "")
        if "text" in fact and not text:
            text = fact["text"]
        tags = fact.get("tags", ["tool_stored"])
        triplets = fact.get("triplets", [])

        if not text:
            return

        if not triplets:
            print(f"[Orchestrator] Extracting triplets for fact: {text}")
            triplets = extract_triplets(text)
        
        print(f"[Orchestrator] Extracted {len(triplets)} triplets")

        added = graph_manager.add_triplets(
            triplets=triplets,
            topic=topic,
            tags=tags,
            fact_text=text,
            user_name=self.user_name,
            emb_model=self.emb_model
        )
        print(f"[Orchestrator] Graph updated via Triplets. Added {added} edges.")

    def _process_update(self, fact: dict, topic: str):
        from src.memory.context_store import delete_fact
        old_fact = fact.get("old_fact", "")
        new_fact = fact.get("new_fact", "")
        tags = fact.get("tags", ["tool_stored"])
        if old_fact and new_fact:
            delete_fact(old_fact, self.emb_model, self.user_name)
            self._process_store({"fact": new_fact, "tags": tags}, topic)

    def _process_delete(self, fact: dict):
        from src.memory.context_store import delete_fact
        text = fact.get("text", "")
        if text:
            delete_fact(text, self.emb_model, self.user_name)
