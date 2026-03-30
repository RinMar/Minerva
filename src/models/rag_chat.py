"""
RAG-enabled chat interface.
Provides multi-turn chat persistence, active memory capabilities, and native XML `<tool>` parsing
to intercept and execute tools (like `retrieve` and `manage_memory`) mid-stream.
"""
import json_repair

from src.models.base_llm import CustomLLM
from src.models.embeddings import get_embedding_model, get_reranker_model
from src.config import PROMPTS
from src.models.tools import execute_retrieve, execute_manage_memory


class RAGChat:
    """
    RAGChat extends a base conversational LLM to provide active memory capabilities.
    It hooks into the generation lifecycle, actively parsing `<tool>` tags, and
    executing `retrieve` and memory tools natively in real-time.
    """
    def __init__(self, user_id, llm=None):
        if llm is None:
            llm = CustomLLM()

        self.llm = llm
        self.history = []
        self.user_id = user_id
        self._emb_model = None
        self._cross_encoder = None

    @property
    def emb_model(self):
        if self._emb_model is None:
            self._emb_model = get_embedding_model()
        return self._emb_model

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = get_reranker_model()
        return self._cross_encoder

    # --- Generative and Tool Interception Core ---

    def _prepare_messages(self, user_prompt: str, system_prompt: str, user_name: str, user_information: dict = None):
        # user_information is fetched via `retrieve` tool dynamically.
        system_message = (
            f"{system_prompt}\n"
            f"You are talking to a user named {user_name}.\n"
        )

        return (
            [{"role": "system", "content": system_message}]
            + self.history
            + [{"role": "user", "content": user_prompt}]
        )

    def _post_generate(self, user_prompt: str, assistant_response: str):
        """Hook for executing logic after a full response is available."""
        pass

    def _handle_parsed_tool(
        self, tool_json_str: str, full_response: str, current_messages: list, override_result: str = None
    ):
        """Executes a parsed tool and appends the result to history and messages."""
        if override_result:
            result_str = override_result
            print(f"\n[❌ Tool Aborted]: {result_str}\n")
        else:
            try:
                parsed = json_repair.loads(tool_json_str)
                tool_name = parsed.get("name")
                tool_args = parsed.get("arguments", {})
                print(f"\n[🔧 Executing Tool: {tool_name}] args: {tool_args}")
                result_str = self._execute_tool(tool_name, tool_args)
                print(f"[✅ Tool Result]: {result_str}\n")
            except Exception as e:
                result_str = f"Tool execution failed: {e}"
                print(f"\n[❌ Tool Error]: {result_str}\n")

        assistant_msg = full_response + "<tool>\n" + tool_json_str + "\n</tool>"
        self.history.append({"role": "assistant", "content": assistant_msg})
        current_messages.append({"role": "assistant", "content": assistant_msg})

        tool_resp_msg = (
            f"Tool execution result:\n<tool_response>\n{result_str}\n"
            f"</tool_response>\nPlease continue your response."
        )
        self.history.append({"role": "user", "content": tool_resp_msg})
        current_messages.append({"role": "user", "content": tool_resp_msg})

    def _process_text_token(self, token: str, buffer: str):
        """
        Process a text token and detect <tool> tags.
        Returns: (content_to_yield, new_buffer, in_tool)
        """
        buffer += token
        if "<tool>" in buffer:
            pre, _, post = buffer.partition("<tool>")
            return pre, post, True

        for s in ["<tool", "<too", "<to", "<t", "<"]:
            if buffer.endswith(s):
                return buffer[:-len(s)], s, False

        return buffer, "", False

    def _stream_generation(self, messages: list, user_prompt: str, max_tokens: int, **kwargs):
        """Internal generator for producing a streaming response and intercepting tool calls."""
        current_messages = list(messages)
        tool_call_count = 0
        MAX_TOOL_CALLS = 5

        while True:
            full_response = ""
            buffer = ""
            in_tool = False
            tool_content = ""

            for token in self.llm.generate(
                messages=current_messages, max_tokens=max_tokens, stream=True, **kwargs
            ):
                if in_tool:
                    tool_content += token
                    if "</tool>" in tool_content:
                        pre, _, _ = tool_content.partition("</tool>")

                        tool_call_count += 1
                        if tool_call_count > MAX_TOOL_CALLS:
                            error_msg = (
                                "Error: Maximum tool call limit reached (5). "
                                "You MUST formulate a final response based on the current context."
                            )
                            self._handle_parsed_tool(
                                pre, full_response, current_messages, override_result=error_msg
                            )
                        else:
                            # We parse and execute the tool, then feed the response back implicitly
                            self._handle_parsed_tool(pre, full_response, current_messages)
                        yield "</action:call>"
                        break
                else:
                    yield_str, buffer, in_tool = self._process_text_token(token, buffer)
                    if yield_str:
                        full_response += yield_str
                        yield yield_str
                    if in_tool:
                        yield "<action:call>"
                        tool_content = buffer
                        buffer = ""
            else:
                # Full token stream exhausted without breaking -> finished response
                self.history.append({"role": "assistant", "content": full_response})
                self._post_generate(user_prompt, full_response)
                return

    def generate(self,
                 user_prompt: str,
                 system_prompt: str = PROMPTS["chat_prompt"],
                 user_name: str = "User",
                 user_information: dict = None,
                 max_tokens: int = 8192,
                 stream: bool = False,
                 **kwargs):
        """
        Streams a response from the LLM. Intercepts `<tool>...</tool>` outputs natively,
        executes the parsed tool via `_execute_tool`, injects the explicit result back
        into the context, and continually streams until the model is satisfied.
        """
        messages = self._prepare_messages(user_prompt, system_prompt, user_name, user_information)
        self.history.append({"role": "user", "content": user_prompt})

        if stream:
            return self._stream_generation(messages, user_prompt, max_tokens, **kwargs)
        else:
            return "Streaming is required for tool usage."

    # --- Tool Execution Core ---

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Executes an invoked tool natively in real-time via external utility routines."""
        if tool_name == "retrieve":
            return execute_retrieve(tool_args, self.user_id, self.emb_model, self.cross_encoder)
        elif tool_name == "manage_memory":
            return execute_manage_memory(tool_args, self.user_id, self.emb_model)
        return f"Error: Tool '{tool_name}' not found."
