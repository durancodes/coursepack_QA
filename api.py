from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import time

from rag import hybrid_search, rerank, generate_answer, process_document

app = FastAPI(title="CoursePack RAG API")

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# REQUEST MODEL
# -------------------------------
class Query(BaseModel):
    question: str

# -------------------------------
# HOME
# -------------------------------
@app.get("/")
def home():
    return {"status": "RAG API running 🚀"}

# -------------------------------
# UPLOAD FILE
# -------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF allowed")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        process_document(file_path)

        return {"message": "File uploaded & processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# ASK QUESTION
# -------------------------------
@app.post("/ask")
def ask_question(query: Query):
    try:
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="Empty question")

        start = time.time()

        docs = hybrid_search(query.question, k=4)
        docs = rerank(query.question, docs, top_k=2)

        answer = generate_answer(query.question, docs)

        end = time.time()

        return {
            "question": query.question,
            "answer": answer,
            "response_time": round(end - start, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))