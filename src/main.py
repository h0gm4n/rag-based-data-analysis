from load_data import load_data
from text_transform import dataframe_to_texts
from aggregation import (
    monthly_sales_summary,
    category_summary,
    regional_summary,
    top_products
)
from chunking import chunk_texts
from embeddings import get_embeddings
from vector_store import create_vector_store
from rag_pipeline import ask_question
from langchain.llms import Ollama


def main():
    df = load_data("data/superstore.csv")

    # Raw row-level texts
    texts = dataframe_to_texts(df)

    # Aggregated texts (IMPORTANT)
    texts += monthly_sales_summary(df)
    texts += category_summary(df)
    texts += regional_summary(df)
    texts += top_products(df)

    docs = chunk_texts(texts)

    embeddings = get_embeddings()
    db = create_vector_store(docs, embeddings)

    llm = Ollama(model="phi3")

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query == "exit":
            break

        answer = ask_question(db, llm, query)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
