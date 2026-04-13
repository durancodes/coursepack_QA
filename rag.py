import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq

# -------------------------------
# GLOBALS
# -------------------------------
chunks = []
db = None
bm25 = None

# -------------------------------
# BETTER EMBEDDINGS (ACCURATE)
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# RERANKER (VERY IMPORTANT)
# -------------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------
# GROQ CLIENT
# -------------------------------
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

    db = FAISS.from_documents(chunks, embeddings)

    texts = [c.page_content for c in chunks]
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

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
def hybrid_search(query, k=3):
    if db is None:
        return []

    vector_results = db.similarity_search(query, k=k)
    keyword_results = bm25_search(query, k=k)

    combined = vector_results + keyword_results
    unique_docs = list({doc.page_content: doc for doc in combined}.values())

    return unique_docs

# -------------------------------
# RERANK (HIGH IMPACT)
# -------------------------------
def rerank(query, docs, top_k=2):
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# -------------------------------
# GENERATE ANSWER (IMPROVED)
# -------------------------------
def generate_answer(query, docs):
    if not docs:
        return "No relevant context found. Please upload documents."

    context = "\n\n".join([doc.page_content[:500] for doc in docs])

    prompt = f"""
You are an expert tutor helping students understand concepts clearly.

Instructions:
- Answer ONLY using the given context
- Explain in a SIMPLE and EASY way (like teaching a student)
- If the concept is technical, break it into small steps
- Use short paragraphs or bullet points when helpful
- Include examples if possible (from the context)
- Do NOT add information outside the context
- If answer is partially available, explain what is available clearly
- If completely not available, say: "This is not clearly mentioned in the document"

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content