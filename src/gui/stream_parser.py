"""
Stream parsing state machine.
Used to decouple string interception logic (e.g. `<think>` and `<action:TOOL>` tokens)
away from the core PySide6 Graphical logic.
"""
import re


class StreamParser:
    def __init__(self, on_token, on_state):
        self.on_token = on_token
        self.on_state = on_state
        self.buffer = ""
        self.is_thinking = False
        self.strip_leading_whitespace = True

    def process_token(self, token: str):
        self.buffer += token

        while self.buffer:
            if self.strip_leading_whitespace and not self.is_thinking:
                self.buffer = self.buffer.lstrip()
                if not self.buffer:
                    break
                self.strip_leading_whitespace = False

            new_buffer, new_is_thinking, break_loop = self._parse_buffer_chunk(self.buffer, self.is_thinking)

            if self.is_thinking and not new_is_thinking:
                self.strip_leading_whitespace = True

            self.is_thinking = new_is_thinking
            self.buffer = new_buffer

            if break_loop:
                break

    def flush(self):
        if self.buffer and not self.is_thinking:
            if self.strip_leading_whitespace:
                self.buffer = self.buffer.lstrip()
            if self.buffer:
                self.on_token(self.buffer)

    def _parse_buffer_chunk(self, buffer, is_thinking):
        """Parse a chunk of the buffer according to the current state."""
        if is_thinking:
            return self._parse_thinking_chunk(buffer)
        return self._parse_normal_chunk(buffer)

    def _parse_normal_chunk(self, buffer):
        """Parse a chunk when not in thinking mode."""
        # Handle <think> -> enter thinking mode
        if "<think>" in buffer:
            return self._handle_tag(buffer, "<think>", "think_start", True)

        # Handle stray </think> in non-thinking mode (silently consume)
        if "</think>" in buffer:
            return self._handle_tag(buffer, "</think>", None, False)

        # Handle <action:TOOL> start tags
        match = re.search(r'<action:([^>]+)>', buffer)
        if match:
            return self._handle_regex_tag(buffer, match, f"action_start_{match.group(1)}", False)

        # Handle </action:TOOL> end tags
        match_end = re.search(r'</action:([^>]+)>', buffer)
        if match_end:
            return self._handle_regex_tag(buffer, match_end, "action_end", False)

        return self._handle_normal_content(buffer)

    def _handle_tag(self, buffer, tag, state_event, new_is_thinking):
        """Helper to handle exact string tags."""
        pre, post = buffer.split(tag, 1)
        if pre:
            self.on_token(pre)
        if state_event:
            self.on_state(state_event)
        return post, new_is_thinking, False

    def _handle_regex_tag(self, buffer, match, state_event, new_is_thinking):
        """Helper to handle regex-based tags."""
        pre = buffer[:match.start()]
        if pre:
            self.on_token(pre)
        if state_event:
            self.on_state(state_event)
        return buffer[match.end():], new_is_thinking, False

    def _handle_normal_content(self, buffer):
        """Handle content that contains partial tags or normal text."""
        if "<" not in buffer:
            self.on_token(buffer)
            return "", False, False

        last_lt = buffer.rfind("<")
        prefix = buffer[last_lt:]

        partial_tags = ["<think>", "</think>", "<action:", "</action:"]
        is_partial = any(
            t.startswith(prefix) and len(prefix) < len(t)
            for t in partial_tags
        )

        if is_partial:
            safe_part = buffer[:last_lt]
            if safe_part:
                self.on_token(safe_part)
            return prefix, False, True

        self.on_token(buffer)
        return "", False, False

    def _parse_thinking_chunk(self, buffer):
        """Parse a chunk when in thinking mode."""
        if "</think>" in buffer:
            _, post = buffer.split("</think>", 1)
            self.on_state("think_end")
            return post, False, False

        if "<" in buffer:
            last_lt = buffer.rfind("<")
            prefix = buffer[last_lt:]
            if "</think>".startswith(prefix) and len(prefix) < len("</think>"):
                return prefix, True, True

        return "", True, False
