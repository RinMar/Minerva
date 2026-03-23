from src.models.rag_chat import RAGChat
from src.memory.db import init_db


class Chat(RAGChat):
    """
    Public API wrapper for RAGChat to support simple usage:
    assistant = Chat()
    for token in assistant.send_message("Hello!", stream=True): ...
    """
    def send_message(self, message: str, stream: bool = True):
        return self.generate(message, stream=stream)


def main():
    init_db()
    chat = RAGChat()

    initial_assistant_response = (
        "Hello, I'm Minerva, your assistant in research, "
        "coding, and more. How can I help you today?"
    )

    print(f"Minerva: {initial_assistant_response}")
    chat.history.append({"role": "assistant", "content": initial_assistant_response})

    while True:
        try:
            user_input = input("You: ")

            if not user_input.strip():
                continue

            print("Minerva: ", end="", flush=True)

            # RAGChat handles intercepting tools and resolving them
            for token in chat.generate(user_input, stream=True):
                print(token, end="", flush=True)

            print()

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
