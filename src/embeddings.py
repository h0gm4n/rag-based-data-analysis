from langchain.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"