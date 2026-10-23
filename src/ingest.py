import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = os.path.join("..", "data", "enterprise_policies.txt")
CHROMA_DIR = "../chroma_db"

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "enterprise_policies.txt")
    chroma_dir = os.path.join(base_dir, "chroma_db")
    loader = TextLoader(data_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=chroma_dir,
        collection_name="enterprise_policies",
    )
    db.persist()
    print(f"Ingested {len(chunks)} chunks into {chroma_dir}")


if __name__ == "__main__":
    main()
