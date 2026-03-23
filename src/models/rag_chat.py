"""
RAG-enabled chat interface.
Used to extend the base conversational LLM to provide active memory capabilities,
intercepting tool calls mid-stream to fetch and store knowledge natively.
"""
from src.models.base_llm import CustomLLM
from src.models.chat_llm import ChatLlm
from src.models.embeddings import embedding_model, reranker_model
from src.memory.context_store import retrieve_context
from src.memory.orchestrator import MemoryOrchestrator


class RAGChat(ChatLlm):
    """
    RAGChat extends the base conversational LLM to provide active memory capabilities.
    It hooks into the generation lifecycle, actively executing `retrieve` and `store`
    tools formulated by the LLM. It intercepts tool calls mid-stream, silently fetches
    knowledge, and returns the context so the model can seamlessly respond to the user.
    """
    def __init__(self, user_name="user", llm=None):
        # Create one shared LLM if not provided
        if llm is None:
            llm = CustomLLM()

        super().__init__(llm=llm)
        self.user_name = user_name

        self.emb_model = embedding_model
        self.cross_encoder = reranker_model

        self.orchestrator = MemoryOrchestrator(
            llm=self.llm, emb_model=self.emb_model, user_name=user_name, cross_encoder=self.cross_encoder
        )

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        Executes an invoked tool natively.

        Supported tools:
        - 'retrieve': Runs a semantic search across the user's persistent knowledge base and graph.
        - 'store': Flags a completely new fact to be parsed and written to disk asynchronously.
        - 'update_fact': Flags an outdated fact to be replaced by a new fact asynchronously.
        - 'delete_fact': Flags a fact for complete deletion asynchronously.

        Returns:
            str: The output of the tool, to be injected back into the LLM context as a system message.
        """
        if tool_name == "retrieve":
            return self._handle_retrieve(tool_args)
        elif tool_name == "manage_memory":
            return self._handle_manage_memory(tool_args)
        return f"Error: Tool '{tool_name}' not found."

    def _handle_retrieve(self, tool_args: dict) -> str:
        query = tool_args.get("query", "")
        if not query:
            return "Error: No query provided."

        search_emb = self.emb_model.encode(query).tolist()
        context = retrieve_context(
            query_emb=search_emb,
            user_name=self.user_name,
            cross_encoder=self.cross_encoder,
            query_text=query,
        )

        if context:
            return f"Retrieved Context:\n{context}"
        return "No relevant context found."

    def _handle_manage_memory(self, tool_args: dict) -> str:
        stores = tool_args.get("store", [])
        updates = tool_args.get("update", [])
        deletes = tool_args.get("delete", [])

        count = 0
        for fact_obj in stores:
            topic = fact_obj.get("topic", "general")
            fact = fact_obj.get("fact", "")
            if fact:
                self.orchestrator.trigger_store(topic, fact)
                count += 1

        for fact_obj in updates:
            topic = fact_obj.get("topic", "general")
            old_fact = fact_obj.get("old_fact", "")
            new_fact = fact_obj.get("new_fact", "")
            if old_fact and new_fact:
                self.orchestrator.trigger_update_fact(topic, old_fact, new_fact)
                count += 1

        for fact_str in deletes:
            if fact_str:
                self.orchestrator.trigger_delete_fact(fact_str)
                count += 1

        return f"Successfully scheduled {count} memory operations (store/update/delete) for processing."

    def _post_generate(self, user_prompt: str, assistant_response: str):
        """Trigger background storage at the end of the turn."""
        self.orchestrator.process_pending_stores()
