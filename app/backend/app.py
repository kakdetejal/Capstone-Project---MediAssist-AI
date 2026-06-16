from fastapi import FastAPI
from pydantic import BaseModel

from rag import retrieve_docs

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "MediAssist AI Backend Running"}


@app.post("/chat")
def chat(request: QueryRequest):

    results = retrieve_docs(
        request.question
    )

    response = []

    for doc, score in results:

        response.append({
            "content": doc.page_content,
            "score": float(score)
        })

    return {
        "question": request.question,
        "results": response
    }