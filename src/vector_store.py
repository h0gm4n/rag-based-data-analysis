from langchain.vectorstores import Chroma

def create_vector_store(docs, embeddings):
    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    db.persist()
    return db