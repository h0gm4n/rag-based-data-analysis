from load_data import load_data
from text_transform import dataframe_to_texts
from aggregation import (
    monthly_sales_summary,
    category_summary,
    regional_summary,
    product_performance,
    discount_analysis,
    global_summary,
    business_insights,
    seasonality_analysis,
    subcategory_profit_margins,
    discount_frequency_analysis,
    regional_performance_analysis,
    state_performance_comparison,
    city_performance_analysis,
    category_trends_comparison,
    region_profit_comparison
)
from chunking import chunk_texts
from embeddings import get_embeddings
from vector_store import create_vector_store
from rag_pipeline import ask_question
from langchain_ollama import OllamaLLM
import polars as pl


def main():
    df = load_data("data/superstore.csv")

    # No raw row-level texts
    texts = []

    # Aggregated texts (IMPORTANT)
    texts += monthly_sales_summary(df)
    texts += seasonality_analysis(df)
    texts += subcategory_profit_margins(df)
    texts += discount_frequency_analysis(df)
    texts += regional_performance_analysis(df)
    texts += region_profit_comparison(df)
    texts += state_performance_comparison(df)
    texts += city_performance_analysis(df)
    texts += category_trends_comparison(df)
    texts += category_summary(df)
    texts += product_performance(df)
    texts += discount_analysis(df)
    texts += global_summary(df)
    texts += business_insights(df)

    docs = chunk_texts(texts)

    embeddings = get_embeddings()
    db = create_vector_store(docs, embeddings)

    llm = OllamaLLM(model="phi3")

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query == "exit":
            break

        answer = ask_question(db, llm, query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
