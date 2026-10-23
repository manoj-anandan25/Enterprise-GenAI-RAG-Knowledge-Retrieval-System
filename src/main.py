import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "../chroma_db"

app = FastAPI(title="Enterprise GenAI & RAG Knowledge Retrieval System")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_documents(request: QueryRequest):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="enterprise_policies",
        )
        results = db.similarity_search_with_score(request.query, k=3)

        answer_text = "\n".join([doc.page_content for doc, _ in results]) if results else ""
        citations = [doc.metadata.get("source", "unknown") for doc, _ in results]
        scores = [float(score) for _, score in results]

        return {
            "query": request.query,
            "answer": answer_text,
            "sources": citations,
            "similarity_scores": scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
