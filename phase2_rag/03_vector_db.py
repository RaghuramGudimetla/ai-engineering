from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "banking_policies"
VECTOR_SIZE = 384

def create_collection():
    print("=" * 60)
    print("Step 1 - Creating Qdrant Collection")
    print("=" * 60)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection : {COLLECTION_NAME}")
    print(f"Vector size        : {VECTOR_SIZE}")
    print(f"Distance metric    : Cosine")


def index_documents():
    print("Step 2 - Indexing Banking Policy Documents")

    documents = [
        {
            "text": "Cash transactions over $10,000 must be reported to the MLRO within 24 hours of detection. A Suspicious Activity Report must be filed immediately.",
            "source": "aml_policy.txt",
            "section": "Suspicious Activity Reporting"
        },
        {
            "text": "Customer Due Diligence must be performed for all new customers before opening an account. CDD involves verifying the identity using government issued documents.",
            "source": "aml_policy.txt",
            "section": "Customer Due Diligence"
        },
        {
            "text": "Enhanced Due Diligence is required for high risk customers, PEPs and customers from high risk jurisdictions. Review every 12 months.",
            "source": "aml_policy.txt",
            "section": "Customer Due Diligence"
        },
        {
            "text": "All CDD documents must be retained for a minimum of 5 years after account closure. Transaction records must be kept for 7 years.",
            "source": "aml_policy.txt",
            "section": "Record Keeping"
        },
        {
            "text": "Fraud alerts scoring above 0.8 must be escalated to a senior fraud analyst immediately. Alerts between 0.5 and 0.8 must be reviewed within 4 hours.",
            "source": "fraud_investigation.txt",
            "section": "Fraud Alert Triage"
        },
        {
            "text": "A card must be blocked immediately when fraud is confirmed. The customer must be notified within 1 hour of card being blocked.",
            "source": "fraud_investigation.txt",
            "section": "Card Blocking Procedures"
        },
        {
            "text": "Provisional credit must be applied to the account within 24 hours of fraud confirmation. Replacement card issued within 3-5 business days.",
            "source": "fraud_investigation.txt",
            "section": "Card Blocking Procedures"
        },
        {
            "text": "High risk customers include Politically Exposed Persons, customers from FATF blacklisted countries, and businesses in high risk sectors like crypto and gambling.",
            "source": "kyc_guidelines.txt",
            "section": "Risk Classification"
        },
        {
            "text": "Source of funds documentation is required for high risk customers. Source of wealth documentation required for accounts over $500,000.",
            "source": "kyc_guidelines.txt",
            "section": "Enhanced Due Diligence"
        },
        {
            "text": "Low risk customers are reviewed every 36 months. Medium risk every 24 months. High risk customers must be reviewed every 12 months.",
            "source": "kyc_guidelines.txt",
            "section": "Risk Classification"
        },
    ]

    points = []
    for doc in documents:
        vector = embedder.encode(doc["text"]).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc["text"],
                    "source": doc["source"],
                    "section": doc["section"]
                }
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} document chunks")


def search(query: str, top_k: int = 3):
    query_vector = embedder.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    return results


def run_searches():
    print("\n" + "=" * 60)
    print("Step 3 - Searching the Vector Database")
    print("=" * 60)

    queries = [
        "What are the rules for reporting suspicious transactions?",
        "How should we handle a confirmed fraud case?",
        "What documents are needed for high risk customers?",
        "How long do we need to keep customer records?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        results = search(query, top_k=2)

        for rank, result in enumerate(results, 1):
            print(f"\nRank {rank} (score: {result.score:.3f})")
            print(f"Source  : {result.payload['source']}")
            print(f"Section : {result.payload['section']}")
            print(f"Text    : {result.payload['text'][:150]}...")


create_collection()
index_documents()
run_searches()

print("\n" + "=" * 60)
print("Key Insight")
print("=" * 60)
print("""
We just built the foundation of RAG:
- Documents chunked and stored as vectors in Qdrant
- Queries converted to vectors and matched by meaning
- Results include metadata (source, section) for citations

Next step: Connect this to Ollama to generate answers
from the retrieved chunks. That is the full RAG pipeline.
""")
