from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------------------------------------
# EMBEDDINGS
# Converting text into vectors (lists of numbers)
# that capture the semantic meaning of the text.
# Similar meaning = similar vectors = close in vector space.
# ------------------------------------------------------------

# Load a small but powerful embedding model
# This runs completely locally - no API needed
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------------------
# EXAMPLE 1 - Basic Embedding
# See what a vector actually looks like
# ------------------------------------------------------------

print("=" * 60)
print("Example 1 - What is a vector?")
print("=" * 60)

text = "Customer due diligence must be performed for all new customers"
vector = model.encode(text)

print(f"Text    : {text}")
print(f"Vector  : {vector[:5]}...")
print(f"Shape   : {vector.shape}")
print(f"Type    : {type(vector)}")
print(f"\nThe full vector has {len(vector)} numbers.")
print("Each number captures a dimension of meaning.")

# ------------------------------------------------------------
# EXAMPLE 2 - Semantic Similarity
# Words that mean the same thing should have similar vectors
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("Example 2 - Semantic Similarity")
print("=" * 60)

pairs = [
    # Similar pairs - should have HIGH similarity
    ("Know Your Customer process", "KYC verification procedure"),
    ("suspicious transaction report", "SAR filing for unusual activity"),
    ("account blocked due to fraud", "card suspended after fraudulent activity"),

    # Different pairs - should have LOW similarity
    ("customer due diligence", "quarterly revenue report"),
    ("fraud alert threshold", "employee annual leave policy"),
]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

for text1, text2 in pairs:
    vec1 = model.encode(text1)
    vec2 = model.encode(text2)
    similarity = cosine_similarity(vec1, vec2)
    label = "SIMILAR" if similarity > 0.5 else "DIFFERENT"
    print(f"\n{label} ({similarity:.3f})")
    print(f"  A: {text1}")
    print(f"  B: {text2}")

# ------------------------------------------------------------
# EXAMPLE 3 - Embedding Banking Policy Chunks
# This is exactly what happens during RAG indexing
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("Example 3 - Embedding Policy Chunks")
print("=" * 60)

chunks = [
    "Cash transactions over $10,000 must be reported to the MLRO within 24 hours.",
    "High risk customers require Enhanced Due Diligence and senior management approval.",
    "Fraud alerts scoring above 0.8 must be escalated to a senior fraud analyst immediately.",
    "KYC documents must be retained for a minimum of 5 years after account closure.",
    "Replacement cards must be issued within 3-5 business days after fraud confirmation.",
]

query = "What is the process for reporting suspicious transactions?"

print(f"Query: {query}")
print(f"\nRanking chunks by relevance...\n")

query_vector = model.encode(query)
chunk_vectors = model.encode(chunks)

scores = []
for i, (chunk, chunk_vec) in enumerate(zip(chunks, chunk_vectors)):
    score = cosine_similarity(query_vector, chunk_vec)
    scores.append((score, chunk))

scores.sort(reverse=True)

for rank, (score, chunk) in enumerate(scores, 1):
    print(f"Rank {rank} (score: {score:.3f})")
    print(f"  {chunk}")
    print()

print("=" * 60)
print("Key Insight")
print("=" * 60)
print("""
The query 'suspicious transactions reporting' correctly ranked
the SAR/MLRO chunk at the top - even though the exact words
do not match. This is semantic search in action.

This is the core of RAG retrieval - find the most relevant
chunks by meaning, not by keyword matching.
""")
