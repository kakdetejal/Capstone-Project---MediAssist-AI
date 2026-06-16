from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def retrieve_docs(query):

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    db = FAISS.load_local(
        "../data/vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = db.similarity_search_with_score(
        query,
        k=5
    )

    return results