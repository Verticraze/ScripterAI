import os, shutil
from typing import List
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from pypdf import PdfReader

app = FastAPI(title="Screenwriter QA")
os.makedirs("data", exist_ok=True)

# --- Embeddings (free) ---
emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = emb_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dim)
chunks: List[str] = []
vecs = None

# --- LLM (free, local) ---
t5_name = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(t5_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
gen = pipeline("text2text-generation", model=llm, tokenizer=tok)

def read_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        text = []
        reader = PdfReader(path)
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "\n".join(text)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(t: str, size=700, overlap=120):
    tokens = t.split()
    out = []
    i = 0
    while i < len(tokens):
        out.append(" ".join(tokens[i:i+size]))
        i += size - overlap
    return [c for c in out if c.strip()]

def embed_and_add(new_chunks: List[str]):
    global vecs
    embs = emb_model.encode(new_chunks, normalize_embeddings=True)
    index.add(embs.astype("float32"))
    if vecs is None: vecs = embs
    else: vecs = np.vstack([vecs, embs])
    return embs

@app.post("/upload")
async def upload(files: List[UploadFile]):
    global chunks, index, vecs
    # reset for a new project session
    index = faiss.IndexFlatIP(dim); chunks = []; vecs = None
    shutil.rmtree("data", ignore_errors=True); os.makedirs("data", exist_ok=True)

    added = 0
    for f in files:
        path = os.path.join("data", f.filename)
        with open(path, "wb") as w: w.write(await f.read())
        text = read_text(path)
        cs = chunk_text(text)
        chunks.extend(cs)
        embed_and_add(cs)
        added += len(cs)
    return {"status": "ok", "chunks": added}

@app.get("/ask")
def ask(q: str, k: int = 5):
    if len(chunks) == 0:
        return JSONResponse({"error":"Upload scripts first."}, status_code=400)
    qv = emb_model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, min(k, len(chunks)))
    ctx = "\n----\n".join(chunks[i] for i in I[0])

    # Limit context length (Flan-T5 max input ~512 tokens)
    MAX_CHARS = 1200   # ~400 tokens (safe margin)
    if len(ctx) > MAX_CHARS:
        ctx = ctx[:MAX_CHARS]

    prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely for a screenwriter:"

    out = gen(prompt, max_new_tokens=256, truncation=True)[0]["generated_text"]
    return {"answer": out, "contexts": [chunks[i] for i in I[0].tolist()]}
