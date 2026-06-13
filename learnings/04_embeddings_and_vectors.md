# Embeddings and Vectors

## What is a Vector?
A vector is a list of numbers that represents the meaning of a piece of text.
Similar meaning = similar numbers = close together in vector space.

Example:
- "Customer due diligence" → [0.23, -0.15, 0.87, 0.42, ...]
- "KYC verification"       → [0.21, -0.14, 0.85, 0.44, ...]  <- very close
- "Quarterly revenue"      → [-0.67, 0.34, -0.21, 0.11, ...] <- far away

## What is an Embedding Model?
A model that converts text into vectors. It runs separately from the LLM.
The embedding model is used for both indexing and querying so they must match.

## What is Vector Size?
How many numbers are used to represent the meaning of a text.
More numbers = more dimensions of meaning captured = more precise matching.

## Embedding Models Comparison
- all-MiniLM-L6-v2    → 384 dimensions, fast, free, local, good for learning
- all-mpnet-base-v2   → 768 dimensions, slower, more accurate, free, local
- text-embedding-3-small → 1536 dimensions, OpenAI, very accurate, costs money
- text-embedding-3-large → 3072 dimensions, OpenAI, best quality, costs more

## Rule of Thumb
- Learning and prototyping    → 384  (all-MiniLM-L6-v2)
- Production internal tools   → 768  (all-mpnet-base-v2)
- Production customer facing  → 1536 (OpenAI text-embedding-3-small)
- Maximum accuracy needed     → 3072 (OpenAI text-embedding-3-large)

## Cosine Similarity
How we measure how close two vectors are.
Returns a value between -1 and 1.
- 1.0  = identical meaning
- 0.8+ = very similar
- 0.5  = somewhat related
- 0.0  = unrelated
- -1.0 = opposite meaning

For RAG retrieval we typically return chunks with similarity above 0.5.

## Why Vectors Enable Semantic Search
Traditional keyword search finds exact word matches.
Vector search finds meaning matches even when words are different.

Example:
Query: "suspicious activity report deadline"
Finds: "SAR must be filed within 24 hours"
Even though none of the words match exactly.

This is the core superpower of RAG over traditional search.
