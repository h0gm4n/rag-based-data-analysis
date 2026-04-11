from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.create_documents(texts)