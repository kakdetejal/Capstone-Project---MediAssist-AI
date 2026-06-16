from pathlib import Path
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_document(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif extension == ".txt":
        loader = TextLoader(file_path)

    else:
        return []

    return loader.load()


def create_vector_store():

    all_docs = []

    data_folder = "../data/raw_data"

    for file in os.listdir(data_folder):

        file_path = os.path.join(data_folder, file)

        docs = load_document(file_path)

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_db.save_local(
        "../data/vector_store"
    )

    print(f"Total Documents Loaded: {len(all_docs)}")
    print(f"Total Chunks Created: {len(chunks)}")
    print("Vector Store Created Successfully")


if __name__ == "__main__":
    create_vector_store()