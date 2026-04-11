def ask_question(db, llm, query):
    retriever = db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(query)

    retrieved_context = "\n\n".join([d.page_content for d in docs])

    global_context = """
    You also have access to overall dataset summaries and trends.
    Use them when answering analytical questions.
    """

    prompt = f"""
    You are a data analyst. Keep answer short and precise.

    {global_context}

    Context:
    {retrieved_context}

    Question: {query}
    """

    return llm.invoke(prompt)