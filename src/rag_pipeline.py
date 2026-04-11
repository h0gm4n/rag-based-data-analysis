def ask_question(db, llm, query: str):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a data analyst.
Answer the question using the context.
Provide concise insights and mention trends if possible.

Context:
{context}

Question: {query}
"""

    return llm.invoke(prompt)
