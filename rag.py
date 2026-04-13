# -------------------------------
# IMPORTS
# -------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from groq import Groq
import os

# -------------------------------
# GLOBALS
# -------------------------------
chunks = []
db = None
bm25 = None

# -------------------------------
# MODELS
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# PROCESS DOCUMENT
# -------------------------------
def process_document(file_path):
    global chunks, db, bm25

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    new_chunks = splitter.split_documents(docs)
    chunks.extend(new_chunks)

    # Rebuild FAISS
    db = FAISS.from_documents(chunks, embeddings)

    # Rebuild BM25
    texts = [chunk.page_content for chunk in chunks]
    tokenized = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized)

    print(f"Total chunks: {len(chunks)}")

# -------------------------------
# BM25 SEARCH
# -------------------------------
def bm25_search(query, k=3):
    if bm25 is None:
        return []

    scores = bm25.get_scores(query.split())
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_n]

# -------------------------------
# HYBRID SEARCH
# -------------------------------
def hybrid_search(query, k=4):
    if db is None:
        return []

    vector_results = db.similarity_search(query, k=k)
    keyword_results = bm25_search(query, k=k)

    combined = vector_results + keyword_results
    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs

# -------------------------------
# RERANK
# -------------------------------
def rerank(query, docs, top_k=2):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# -------------------------------
# GENERATE ANSWER
# -------------------------------
def generate_answer(query, docs):
    if not docs:
        return "No relevant documents found. Upload a document first."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant.

Rules:
- Answer ONLY from context
- If not found, say "I don't know"
- Keep answer short (2-3 lines)

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content