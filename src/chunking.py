from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.create_documents(texts)