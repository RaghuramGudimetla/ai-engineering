from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
import uuid

# ------------------------------------------------------------
# MULTIPLE COLLECTIONS AND ROUTING
# Organise documents into separate collections
# Route questions to the right collection automatically
# ------------------------------------------------------------

client = QdrantClient(url="http://localhost:6333")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = 384

# ------------------------------------------------------------
# DEFINE COLLECTIONS AND THEIR DOCUMENTS
# ------------------------------------------------------------

COLLECTIONS = {
    "aml_policies": {
        "description": "Anti money laundering policies and suspicious activity reporting rules",
        "keywords": ["aml", "suspicious", "money laundering", "SAR", "MLRO", "reporting", "transaction monitoring"],
        "documents": [
            {
                "text": "Cash transactions over $10,000 must be reported to the MLRO within 24 hours of detection.",
                "section": "Suspicious Activity Reporting",
                "version": 3
            },
            {
                "text": "A Suspicious Activity Report must be filed for multiple transactions structured to avoid reporting thresholds.",
                "section": "Suspicious Activity Reporting",
                "version": 3
            },
            {
                "text": "The bank operates automated transaction monitoring that flags unusual patterns compared to customer profile.",
                "section": "Transaction Monitoring",
                "version": 3
            },
            {
                "text": "All flagged transactions must be reviewed within 48 hours by the compliance team.",
                "section": "Transaction Monitoring",
                "version": 3
            },
        ]
    },
    "fraud_runbooks": {
        "description": "Fraud investigation procedures, alert triage and card blocking procedures",
        "keywords": ["fraud", "alert", "card block", "investigation", "triage", "provisional credit"],
        "documents": [
            {
                "text": "Fraud alerts scoring above 0.8 must be escalated to a senior fraud analyst immediately.",
                "section": "Alert Triage",
                "version": 2
            },
            {
                "text": "Alerts between 0.5 and 0.8 must be reviewed within 4 hours by a fraud analyst.",
                "section": "Alert Triage",
                "version": 2
            },
            {
                "text": "A card must be blocked immediately when fraud is confirmed. Customer notified within 1 hour.",
                "section": "Card Blocking",
                "version": 2
            },
            {
                "text": "Provisional credit must be applied to the account within 24 hours of fraud confirmation.",
                "section": "Card Blocking",
                "version": 2
            },
        ]
    },
    "kyc_guidelines": {
        "description": "Know your customer guidelines, risk classification and due diligence requirements",
        "keywords": ["kyc", "know your customer", "due diligence", "risk rating", "PEP", "identity", "verification"],
        "documents": [
            {
                "text": "Customer Due Diligence must be performed for all new customers before opening an account.",
                "section": "CDD Requirements",
                "version": 3
            },
            {
                "text": "Enhanced Due Diligence is required for high risk customers, PEPs and customers from high risk jurisdictions.",
                "section": "Enhanced Due Diligence",
                "version": 3
            },
            {
                "text": "High risk customers include Politically Exposed Persons and customers from FATF blacklisted countries.",
                "section": "Risk Classification",
                "version": 3
            },
            {
                "text": "Low risk customers reviewed every 36 months. Medium risk every 24 months. High risk every 12 months.",
                "section": "Review Cycles",
                "version": 3
            },
        ]
    },
    "loan_policies": {
        "description": "Loan application requirements, approval criteria and documentation needed",
        "keywords": ["loan", "mortgage", "application", "credit", "approval", "income", "bank statements"],
        "documents": [
            {
                "text": "Loan applicants must provide 3 months of bank statements and proof of income.",
                "section": "Application Requirements",
                "version": 1
            },
            {
                "text": "Mortgage applications require a property valuation report from an approved surveyor.",
                "section": "Mortgage Requirements",
                "version": 1
            },
            {
                "text": "Loan approval requires a minimum credit score of 650 and debt to income ratio below 40 percent.",
                "section": "Approval Criteria",
                "version": 1
            },
            {
                "text": "All loan applications must be reviewed by a senior credit officer for amounts above $500,000.",
                "section": "Approval Criteria",
                "version": 1
            },
        ]
    }
}


# ------------------------------------------------------------
# STEP 1 - Setup all collections
# ------------------------------------------------------------

def setup_collections():
    print("=" * 60)
    print("Setting up collections")
    print("=" * 60)

    existing = [c.name for c in client.get_collections().collections]

    for collection_name, config in COLLECTIONS.items():
        if collection_name in existing:
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        points = []
        for doc in config["documents"]:
            vector = embedder.encode(doc["text"]).tolist()
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        **doc,
                        "collection": collection_name
                    }
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        print(f"  {collection_name}: {len(points)} chunks indexed")

    print(f"\nTotal collections: {len(COLLECTIONS)}")


# ------------------------------------------------------------
# STEP 2 - Smart Router
# Uses the LLM to decide which collection to search
# ------------------------------------------------------------

def route_question(question: str) -> list[str]:
    collection_descriptions = "\n".join([
        f"- {name}: {config['description']}"
        for name, config in COLLECTIONS.items()
    ])

    prompt = f"""You are a routing assistant for a banking document system.
Given a user question, decide which document collections to search.

Available collections:
{collection_descriptions}

Question: {question}

Reply with ONLY the collection names to search, comma separated.
Choose one collection for specific questions.
Choose multiple collections for broad questions.
Example reply: aml_policies
Example reply: fraud_runbooks, kyc_guidelines

Collections to search:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse the response to get collection names
    response_text = response.message.content.strip().lower()
    selected = []

    for collection_name in COLLECTIONS.keys():
        if collection_name in response_text:
            selected.append(collection_name)

    # Fallback - search all if router fails
    if not selected:
        selected = list(COLLECTIONS.keys())

    return selected


# ------------------------------------------------------------
# STEP 3 - Search specific collections
# ------------------------------------------------------------

def search_collections(question: str, collections: list[str], top_k: int = 2) -> list[dict]:
    query_vector = embedder.encode(question).tolist()
    all_results = []

    for collection_name in collections:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points

        for result in results:
            all_results.append({
                "text": result.payload["text"],
                "section": result.payload["section"],
                "collection": collection_name,
                "score": result.score
            })

    # Sort all results by score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k * len(collections)]


# ------------------------------------------------------------
# STEP 4 - Generate answer from retrieved chunks
# ------------------------------------------------------------

def generate_answer(question: str, chunks: list[dict]):
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}: {chunk['collection']} - {chunk['section']}]\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""You are a Banking Compliance Assistant.
Answer using ONLY the context below.
If answer not found say: I could not find this in the policy documents.
Always cite the source collection and section.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response:
        print(chunk.message.content, end="", flush=True)
    print()


# ------------------------------------------------------------
# STEP 5 - Full routed RAG pipeline
# ------------------------------------------------------------

def routed_rag(question: str):
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print("=" * 60)

    # Route the question
    collections = route_question(question)
    print(f"Router selected: {collections}")

    # Search selected collections
    chunks = search_collections(question, collections)
    print(f"Retrieved {len(chunks)} chunks\n")

    # Show retrieved chunks
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i} [{chunk['score']:.3f}] {chunk['collection']} - {chunk['section']}")

    print("\nAnswer:")
    generate_answer(question, chunks)


# ------------------------------------------------------------
# RUN DEMOS
# ------------------------------------------------------------

setup_collections()

questions = [
    "What is the threshold for reporting suspicious transactions?",
    "How do I handle a fraud alert with score 0.9?",
    "What documents does a loan applicant need to provide?",
    "What are the KYC requirements for high risk customers?",
    "What are the rules around customer due diligence and fraud reporting?",
]

for question in questions:
    routed_rag(question)

print("\n" + "=" * 60)
print("Key Insight")
print("=" * 60)
print("""
Single collection   → fast, precise, user knows what they want
Multiple collections→ broader, catches cross domain questions
Smart routing       → best of both, LLM decides automatically

In production routing can be done by:
- LLM (what we did here - flexible but adds latency)
- Keywords (fast but rigid)
- User selection (let user pick the domain)
- ML classifier (trained router, fast and smart)
""")
