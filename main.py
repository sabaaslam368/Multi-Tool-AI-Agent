
from agent import ask_agent


def main():
    print("=" * 60)
    print("          MULTI-TOOL AI RESEARCH ASSISTANT")
    print("=" * 60)

    print("\nAvailable tools:")
    print("1. Calculator")
    print("2. Research file reader")
    print("3. Research notes search")
    print("4. AI/CS study helper")

    print("\nExample questions:")
    print("- Calculate 125 * 24.")
    print("- Read research_notes.txt and explain RAG.")
    print("- Search my research notes for embeddings.")
    print("- Explain LangGraph and calculate 25 * 18.")
    print("\nType 'exit' to close the application.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            print("\nAgent is working...")

            answer = ask_agent(question)

            print("\nAssistant:", answer)

        except KeyboardInterrupt:
            print("\nRequest cancelled.")
            break

        except Exception as error:
            print("\nThe request could not be completed.")
            print(
                "Check your API key, internet connection, "
                "model availability, and tool configuration."
            )
            print("Technical details:", error)


if __name__ == "__main__":
    main()
