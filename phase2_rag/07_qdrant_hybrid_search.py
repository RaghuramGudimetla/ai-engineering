from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector, NamedVector, NamedSparseVector,
    Prefetch, FusionQuery, Fusion
)
import uuid

# ------------------------------------------------------------
# HYBRID SEARCH
# Combines dense vector search (semantic) with
# sparse vector search (keyword/BM25)
# Best of both worlds
# ------------------------------------------------------------

client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "banking_hybrid"
VECTOR_SIZE = 384

# ------------------------------------------------------------
# BM25 - The keyword search algorithm
# Scores documents based on term frequency
# Similar to how search engines work
# ------------------------------------------------------------

def bm25_sparse_vector(text: str, vocabulary: dict) -> SparseVector:
    """
    Create a sparse vector from text using simple term frequency.
    In production you would use a proper BM25 library.
    Here we simulate it with term frequency scoring.
    """
    import math

    words = text.lower().split()
    term_freq = {}
    for word in words:
        # Remove punctuation
        word = word.strip(".,!?;:()")
        if len(word) > 2:  # ignore very short words
            term_freq[word] = term_freq.get(word, 0) + 1

    indices = []
    values = []

    for word, freq in term_freq.items():
        if word in vocabulary:
            idx = vocabulary[word]
            # Simple TF score
            tf_score = 1 + math.log(freq)
            indices.append(idx)
            values.append(tf_score)

    if not indices:
        indices = [0]
        values = [0.0]

    return SparseVector(indices=indices, values=values)


def build_vocabulary(documents: list[str]) -> dict:
    """Build a vocabulary from all documents"""
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            word = word.strip(".,!?;:()")
            if len(word) > 2 and word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------

def setup():
    print("=" * 60)
    print("Setting up Hybrid Search Collection")
    print("=" * 60)

    documents = [
        {
            "text": "Cash transactions over $10,000 must be reported to the MLRO within 24 hours.",
            "source": "aml_policy.txt",
            "section": "Suspicious Activity Reporting"
        },
        {
            "text": "SAR filing is mandatory for structured transactions designed to avoid the $10,000 threshold.",
            "source": "aml_policy.txt",
            "section": "SAR Requirements"
        },
        {
            "text": "Customer Due Diligence must be performed before opening any new account.",
            "source": "kyc_guidelines.txt",
            "section": "CDD Requirements"
        },
        {
            "text": "Enhanced Due Diligence required for PEPs and FATF blacklisted country nationals.",
            "source": "kyc_guidelines.txt",
            "section": "Enhanced Due Diligence"
        },
        {
            "text": "Fraud alerts with score above 0.8 escalated to senior analyst immediately.",
            "source": "fraud_runbook.txt",
            "section": "Alert Triage"
        },
        {
            "text": "Basel III compliance requires minimum capital adequacy ratio of 8 percent.",
            "source": "regulatory.txt",
            "section": "Basel III Requirements"
        },
        {
            "text": "FATF Recommendation 16 requires wire transfer records to include originator information.",
            "source": "regulatory.txt",
            "section": "FATF Requirements"
        },
        {
            "text": "Visa Platinum cardholders are eligible for provisional credit within 24 hours of fraud report.",
            "source": "product_policy.txt",
            "section": "Visa Platinum Benefits"
        },
        {
            "text": "Home Loan Plus product requires minimum credit score of 700 and 20 percent deposit.",
            "source": "product_policy.txt",
            "section": "Home Loan Plus Requirements"
        },
        {
            "text": "AML CTF program must be reviewed annually by the board of directors.",
            "source": "aml_policy.txt",
            "section": "Program Review"
        },
    ]

    # Build vocabulary for sparse vectors
    all_texts = [doc["text"] for doc in documents]
    vocabulary = build_vocabulary(all_texts)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    # Create collection with BOTH dense and sparse vectors
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }
    )

    # Index documents with both vector types
    points = []
    for doc in documents:
        dense_vector = embedder.encode(doc["text"]).tolist()
        sparse_vector = bm25_sparse_vector(doc["text"], vocabulary)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                payload=doc
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} documents with dense + sparse vectors")
    return vocabulary


def dense_search(query: str, top_k: int = 5) -> list[dict]:
    query_vector = embedder.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        using="dense",
        query=query_vector,
        limit=top_k
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "section": r.payload["section"],
            "score": r.score,
            "method": "dense"
        }
        for r in results
    ]


def sparse_search(query: str, vocabulary: dict, top_k: int = 5) -> list[dict]:
    sparse_vector = bm25_sparse_vector(query, vocabulary)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        using="sparse",
        query=sparse_vector,
        limit=top_k
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "section": r.payload["section"],
            "score": r.score,
            "method": "sparse"
        }
        for r in results
    ]


def hybrid_search(query: str, vocabulary: dict, top_k: int = 5) -> list[dict]:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF)
    Combines dense and sparse results into one ranked list
    """
    dense_vector = embedder.encode(query).tolist()
    sparse_vector = bm25_sparse_vector(query, vocabulary)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=top_k),
            Prefetch(query=sparse_vector, using="sparse", limit=top_k)
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k
    ).points

    return [
        {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "section": r.payload["section"],
            "score": r.score,
            "method": "hybrid"
        }
        for r in results
    ]


def compare_search_methods(query: str, vocabulary: dict):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("=" * 60)

    dense  = dense_search(query, top_k=3)
    sparse = sparse_search(query, vocabulary, top_k=3)
    hybrid = hybrid_search(query, vocabulary, top_k=3)

    print("\nDense (semantic) results:")
    print("-" * 40)
    for i, r in enumerate(dense, 1):
        print(f"  {i}. [{r['score']:.3f}] {r['section']}")
        print(f"      {r['text'][:80]}...")

    print("\nSparse (keyword) results:")
    print("-" * 40)
    for i, r in enumerate(sparse, 1):
        print(f"  {i}. [{r['score']:.3f}] {r['section']}")
        print(f"      {r['text'][:80]}...")

    print("\nHybrid (RRF combined) results:")
    print("-" * 40)
    for i, r in enumerate(hybrid, 1):
        print(f"  {i}. [{r['score']:.3f}] {r['section']}")
        print(f"      {r['text'][:80]}...")


# ------------------------------------------------------------
# RUN DEMOS
# ------------------------------------------------------------

vocabulary = setup()

# Query 1 - semantic query - vector search should win
compare_search_methods(
    "what are the rules around reporting unusual financial activity?",
    vocabulary
)

# Query 2 - exact term query - keyword search should win
compare_search_methods(
    "FATF Recommendation 16 wire transfer",
    vocabulary
)

# Query 3 - mixed query - hybrid should win
compare_search_methods(
    "SAR filing requirements for $10,000 transactions",
    vocabulary
)

print("\n" + "=" * 60)
print("Key Insight")
print("=" * 60)
print("""
Dense search   - wins on meaning and paraphrasing
Sparse search  - wins on exact terms, codes, amounts
Hybrid search  - wins overall by combining both

In production banking RAG use hybrid search because:
- Regulation codes (FATF, Basel III) need exact matching
- Dollar amounts ($10,000) need exact matching
- Policy concepts need semantic matching
- Hybrid covers all cases
""")
