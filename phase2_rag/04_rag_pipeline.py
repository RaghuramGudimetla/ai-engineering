import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os

# ------------------------------------------------------------
# FULL RAG PIPELINE
# 1. Load and chunk documents
# 2. Embed chunks and store in Qdrant
# 3. Take user question
# 4. Retrieve relevant chunks
# 5. Generate answer using Ollama + retrieved chunks
# ------------------------------------------------------------

client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "banking_rag"
VECTOR_SIZE = 384
DOCUMENTS_PATH = "phase2_rag/documents"


# ------------------------------------------------------------
# STEP 1 - Load and Chunk Documents
# ------------------------------------------------------------

def load_and_chunk_documents() -> list[dict]:
    chunks = []

    for filename in os.listdir(DOCUMENTS_PATH):
        if not filename.endswith(".txt"):
            continue

        with open(f"{DOCUMENTS_PATH}/{filename}", "r") as f:
            text = f.read()

        # Section chunking - best for structured policy docs
        lines = text.split("\n")
        current_chunk = []
        current_section = "Introduction"

        for line in lines:
            is_header = (
                line.isupper() and len(line) > 5 or
                line.startswith("SECTION") or
                line.startswith("CHAPTER") or
                line.startswith("INTRODUCTION")
            )

            if is_header and current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if len(chunk_text) > 50:
                    chunks.append({
                        "text": chunk_text,
                        "source": filename,
                        "section": current_section
                    })
                current_chunk = [line]
                current_section = line.strip()
            else:
                current_chunk.append(line)

        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if len(chunk_text) > 50:
                chunks.append({
                    "text": chunk_text,
                    "source": filename,
                    "section": current_section
                })

    return chunks


# ------------------------------------------------------------
# STEP 2 - Index Documents into Qdrant
# ------------------------------------------------------------

def index_documents(chunks: list[dict]):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    points = []
    for chunk in chunks:
        vector = embedder.encode(chunk["text"]).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=chunk
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} chunks from {DOCUMENTS_PATH}")


# ------------------------------------------------------------
# STEP 3 - Retrieve Relevant Chunks
# ------------------------------------------------------------

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_vector = embedder.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "section": r.payload["section"],
            "score": r.score
        }
        for r in results
    ]


# ------------------------------------------------------------
# STEP 4 - Generate Answer Using Retrieved Chunks
# ------------------------------------------------------------

def generate_answer(question: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}: {chunk['source']} - {chunk['section']}]\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""You are a Banking Compliance Assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this in the policy documents."
Always cite which source and section your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    full_response = ""
    for chunk in response:
        print(chunk.message.content, end="", flush=True)
        full_response += chunk.message.content

    return full_response


# ------------------------------------------------------------
# STEP 5 - Full RAG Function
# ------------------------------------------------------------

def rag(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Retrieve
    chunks = retrieve(question, top_k=3)

    print(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. [{chunk['score']:.3f}] {chunk['source']} - {chunk['section']}")

    print("\nAnswer:")
    generate_answer(question, chunks)
    print("\n")


# ------------------------------------------------------------
# MAIN - Index once then answer questions
# ------------------------------------------------------------

print("=" * 60)
print("Banking Policy RAG System")
print("=" * 60)

print("\nLoading and indexing documents...")
chunks = load_and_chunk_documents()
index_documents(chunks)

questions = [
    "What is the process for reporting a suspicious transaction?",
    "What are the requirements for high risk customer reviews?",
    "What happens when a fraud alert scores above 0.8?",
    "How long must KYC documents be retained?",
    "What is needed for source of wealth documentation?",
]

for question in questions:
    rag(question)

print("=" * 60)
print("Key Insight")
print("=" * 60)
print("""
This is a complete RAG pipeline:
- Documents loaded and chunked from real files
- Chunks embedded and stored in Qdrant
- Questions retrieved by semantic similarity
- Answers generated using only retrieved context
- Sources cited so answers are traceable

The LLM is NOT using its training data to answer.
It is only using your documents. This is the core
value of RAG in enterprise applications.
""")
