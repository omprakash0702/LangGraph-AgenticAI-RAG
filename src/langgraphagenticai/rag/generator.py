def rag_generate(state: dict, llm) -> dict:
    docs = state["context"]
    question = state["messages"][-1].content

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer strictly using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    state["rag_answer"] = response.content
    return state
