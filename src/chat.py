"""
Main interactive chat interface for Minerva.
Used to provide the terminal-based REPL for interacting with the assistant,
and to expose the `Chat` wrapper class for API usage.
"""
from src.models.rag_chat import RAGChat
from src.memory.db import init_db, get_session, User


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

    # Ensure default user exists for CLI
    with get_session() as session:
        user = session.query(User).filter_by(name="user").first()
        if not user:
            user = User(name="user")
            session.add(user)
            session.commit()
            user_id = user.id
        else:
            user_id = user.id

    chat = RAGChat(user_id=user_id)

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
