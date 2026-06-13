from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
import uuid

# ------------------------------------------------------------
# QDRANT METADATA FILTERING
# Vector search + filters = precise retrieval
# Think of filters as WHERE clauses on your vector search
# ------------------------------------------------------------

client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "banking_filtering_demo"
VECTOR_SIZE = 384

# ------------------------------------------------------------
# SETUP - Create collection with rich metadata
# ------------------------------------------------------------

def setup():
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

    # Documents with rich metadata - notice more fields than before
    documents = [
        {
            "text": "Cash transactions over $10,000 must be reported to the MLRO within 24 hours.",
            "source": "aml_policy.txt",
            "section": "Suspicious Activity Reporting",
            "doc_type": "policy",
            "jurisdiction": "US",
            "risk_level": "high",
            "policy_version": 3,
            "active": True
        },
        {
            "text": "Customer Due Diligence must be performed for all new customers before opening an account.",
            "source": "aml_policy.txt",
            "section": "Customer Due Diligence",
            "doc_type": "policy",
            "jurisdiction": "US",
            "risk_level": "high",
            "policy_version": 3,
            "active": True
        },
        {
            "text": "Enhanced Due Diligence is required for high risk customers and PEPs.",
            "source": "aml_policy.txt",
            "section": "Customer Due Diligence",
            "doc_type": "policy",
            "jurisdiction": "UK",
            "risk_level": "high",
            "policy_version": 3,
            "active": True
        },
        {
            "text": "All CDD documents must be retained for a minimum of 5 years after account closure.",
            "source": "aml_policy.txt",
            "section": "Record Keeping",
            "doc_type": "policy",
            "jurisdiction": "US",
            "risk_level": "medium",
            "policy_version": 2,
            "active": False  # old version - superseded
        },
        {
            "text": "Fraud alerts scoring above 0.8 must be escalated to a senior fraud analyst immediately.",
            "source": "fraud_investigation.txt",
            "section": "Fraud Alert Triage",
            "doc_type": "runbook",
            "jurisdiction": "US",
            "risk_level": "high",
            "policy_version": 2,
            "active": True
        },
        {
            "text": "A card must be blocked immediately when fraud is confirmed. Customer notified within 1 hour.",
            "source": "fraud_investigation.txt",
            "section": "Card Blocking Procedures",
            "doc_type": "runbook",
            "jurisdiction": "US",
            "risk_level": "high",
            "policy_version": 2,
            "active": True
        },
        {
            "text": "High risk customers include Politically Exposed Persons and customers from FATF blacklisted countries.",
            "source": "kyc_guidelines.txt",
            "section": "Risk Classification",
            "doc_type": "guideline",
            "jurisdiction": "UK",
            "risk_level": "high",
            "policy_version": 3,
            "active": True
        },
        {
            "text": "Low risk customers are reviewed every 36 months. Medium risk every 24 months.",
            "source": "kyc_guidelines.txt",
            "section": "Risk Classification",
            "doc_type": "guideline",
            "jurisdiction": "US",
            "risk_level": "low",
            "policy_version": 3,
            "active": True
        },
        {
            "text": "Source of wealth documentation required for accounts over $500,000.",
            "source": "kyc_guidelines.txt",
            "section": "Enhanced Due Diligence",
            "doc_type": "guideline",
            "jurisdiction": "US",
            "risk_level": "high",
            "policy_version": 3,
            "active": True
        },
        {
            "text": "Loan applicants must provide 3 months of bank statements and proof of income.",
            "source": "loan_policy.txt",
            "section": "Application Requirements",
            "doc_type": "policy",
            "jurisdiction": "US",
            "risk_level": "medium",
            "policy_version": 1,
            "active": True
        },
    ]

    points = []
    for doc in documents:
        vector = embedder.encode(doc["text"]).tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=doc
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} chunks with rich metadata")


def search(query: str, query_filter=None, top_k: int = 3, label: str = ""):
    print(f"\n{'='*60}")
    print(f"Query : {query}")
    if label:
        print(f"Filter: {label}")
    print("=" * 60)

    query_vector = embedder.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k
    ).points

    if not results:
        print("No results found")
        return

    for rank, result in enumerate(results, 1):
        p = result.payload
        print(f"\nRank {rank} (score: {result.score:.3f})")
        print(f"  Source      : {p['source']}")
        print(f"  Section     : {p['section']}")
        print(f"  Doc Type    : {p['doc_type']}")
        print(f"  Jurisdiction: {p['jurisdiction']}")
        print(f"  Risk Level  : {p['risk_level']}")
        print(f"  Version     : {p['policy_version']}")
        print(f"  Active      : {p['active']}")
        print(f"  Text        : {p['text'][:100]}...")


# ------------------------------------------------------------
# FILTER 1 - Match exact value
# Only search within a specific doc type
# SQL equivalent: WHERE doc_type = 'policy'
# ------------------------------------------------------------

def demo_exact_match():
    print("\n\n" + "#" * 60)
    print("# Filter 1 - Exact Match")
    print("# WHERE doc_type = 'policy'")
    print("#" * 60)

    search(
        query="what are the customer due diligence requirements?",
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value="policy")
                )
            ]
        ),
        label="doc_type = policy"
    )


# ------------------------------------------------------------
# FILTER 2 - Multiple conditions (AND)
# Only active UK policies
# SQL equivalent: WHERE jurisdiction = 'UK' AND active = true
# ------------------------------------------------------------

def demo_multiple_conditions():
    print("\n\n" + "#" * 60)
    print("# Filter 2 - Multiple Conditions (AND)")
    print("# WHERE jurisdiction = 'UK' AND active = true")
    print("#" * 60)

    search(
        query="what are the requirements for high risk customers?",
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="jurisdiction",
                    match=MatchValue(value="UK")
                ),
                FieldCondition(
                    key="active",
                    match=MatchValue(value=True)
                )
            ]
        ),
        label="jurisdiction = UK AND active = true"
    )


# ------------------------------------------------------------
# FILTER 3 - Range filter
# Only policies version 3 and above
# SQL equivalent: WHERE policy_version >= 3
# ------------------------------------------------------------

def demo_range_filter():
    print("\n\n" + "#" * 60)
    print("# Filter 3 - Range Filter")
    print("# WHERE policy_version >= 3")
    print("#" * 60)

    search(
        query="what are the reporting requirements?",
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="policy_version",
                    range=Range(gte=3)
                )
            ]
        ),
        label="policy_version >= 3"
    )


# ------------------------------------------------------------
# FILTER 4 - Must NOT (exclude)
# Exclude old inactive policies
# SQL equivalent: WHERE active != false
# ------------------------------------------------------------

def demo_must_not():
    print("\n\n" + "#" * 60)
    print("# Filter 4 - Must NOT (Exclude)")
    print("# WHERE active != false")
    print("#" * 60)

    search(
        query="how long must documents be retained?",
        query_filter=Filter(
            must_not=[
                FieldCondition(
                    key="active",
                    match=MatchValue(value=False)
                )
            ]
        ),
        label="exclude inactive policies"
    )


# ------------------------------------------------------------
# FILTER 5 - High risk only across all doc types
# Most practical for AML investigations
# SQL equivalent: WHERE risk_level = 'high' AND active = true
# ------------------------------------------------------------

def demo_risk_filter():
    print("\n\n" + "#" * 60)
    print("# Filter 5 - High Risk Active Documents Only")
    print("# WHERE risk_level = 'high' AND active = true")
    print("#" * 60)

    search(
        query="what actions must be taken for suspicious activity?",
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="risk_level",
                    match=MatchValue(value="high")
                ),
                FieldCondition(
                    key="active",
                    match=MatchValue(value=True)
                )
            ]
        ),
        label="risk_level = high AND active = true"
    )


# ------------------------------------------------------------
# RUN ALL DEMOS
# ------------------------------------------------------------

print("=" * 60)
print("Qdrant Metadata Filtering Demo")
print("=" * 60)

setup()
demo_exact_match()
demo_multiple_conditions()
demo_range_filter()
demo_must_not()
demo_risk_filter()

print("\n\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
Filter Type          SQL Equivalent              Qdrant
---------------------------------------------------------------
Exact match          WHERE field = value         MatchValue
Multiple AND         WHERE a = x AND b = y       must: [...]
Range                WHERE version >= 3          Range(gte=3)
Exclude              WHERE active != false        must_not: [...]
OR conditions        WHERE a = x OR b = y        should: [...]

Key insight:
Filtering happens INSIDE the vector search - not after.
Qdrant only searches vectors that match the filter.
This makes it fast even with millions of vectors.
""")
