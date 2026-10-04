"""Step 5 - assemble the prompt that grounds the LLM in retrieved context."""

SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know based on the documents. "
    "Be concise and mention the source file and page when useful."
)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits
    ) or "(no documents indexed)"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
