# Vector Databases and Qdrant

## What is a Vector Database?
A database optimised for storing and searching vectors.
Like a regular database but instead of querying by exact value
you query by semantic similarity.

## Vector DB Comparison
- Qdrant     → open source, local or cloud, best filtering, production ready
- ChromaDB   → simplest to start, good for prototyping
- Pinecone   → fully managed cloud, no ops, costs money
- pgvector   → Postgres extension, use if you already have Postgres
- Weaviate   → enterprise scale, multi tenancy, complex setup
- FAISS      → Facebook library, in memory, no persistence, research use

## Qdrant Core Concepts

### Collection
Container for vectors of the same type. Like a table in SQL.
Each collection has a fixed vector size and distance metric.
You must declare vector size upfront and it must match your embedding model.

### Point
A single entry in a collection consisting of:
- id      → unique identifier
- vector  → the embedding numbers
- payload → metadata stored alongside the vector

### Distance Metric
How similarity between vectors is measured.
- Cosine    → measures angle, best for text, most common
- Euclidean → measures straight line distance, good for numerical data
- Dot Product → measures magnitude and direction, used with normalised vectors

For RAG with text always use Cosine.

## Metadata Filtering
Filter which vectors to search before or during similarity search.
Like a SQL WHERE clause on top of your vector search.

### Filter Types
- MatchValue  → exact match           WHERE field = value
- MatchAny    → match from list        WHERE field IN (x, y, z)
- MatchExcept → exclude values         WHERE field NOT IN (x, y)
- MatchText   → keyword in text field  WHERE field LIKE '%text%'
- Range       → numeric comparison     WHERE field >= 3
- must        → all conditions match   AND
- should      → any condition matches  OR
- must_not    → exclude matches        NOT

### Why Filtering Matters
Without filtering RAG might return chunks from wrong document versions,
wrong jurisdictions or completely irrelevant document types.
In banking this is a compliance risk.

### Pre-filtering
Qdrant filters during the vector search not after.
This means it only searches vectors that match the filter.
Much faster at scale than filtering results after retrieval.

## Multiple Collections
In production organise documents into separate collections by domain.

Example banking setup:
- aml_policies    → AML and SAR rules
- fraud_runbooks  → fraud investigation procedures
- kyc_guidelines  → KYC and due diligence rules
- loan_policies   → loan application requirements

Benefits:
- Cleaner organisation
- Different chunking strategies per collection
- Different metadata schemas per collection
- Faster search within a domain
- Easier to update one domain without affecting others

## Collection Routing
Deciding which collection to search based on the user question.
Like a query router in a data warehouse.

Options:
- LLM routing    → flexible, understands complex questions, adds latency
- Keyword routing → fast, rigid, misses paraphrasing
- User selection → let user pick the domain explicitly
- ML classifier  → trained router, fast and smart, needs training data

## Hybrid Search
Combines dense vector search with sparse keyword search.
- Dense search  → good at meaning and paraphrasing
- Sparse search → good at exact terms, codes, amounts
- Hybrid        → good at both

Use Reciprocal Rank Fusion (RRF) to combine rankings from both searches.

In banking use hybrid search because:
- Regulation codes like FATF and Basel III need exact matching
- Dollar amounts like $10,000 need exact matching
- Policy concepts need semantic matching
- Hybrid covers all cases

## Reranking
Two stage retrieval for higher precision.

Stage 1 - Vector search retrieves top 20 candidates (fast, approximate)
Stage 2 - Cross encoder reranks candidates against the question (slow, precise)

The cross encoder reads the question AND the chunk together to score relevance.
Much more accurate than vector similarity alone.

Production pattern:
- Vector search → retrieve top 20 to 50 candidates
- Reranker      → rerank to top 3 to 5
- LLM           → generate answer from reranked chunks

Use reranking for high stakes applications where precision matters.
In banking compliance the extra latency is worth it.

## The Complete RAG Stack
- Embedding model  → converts text to vectors (sentence-transformers)
- Vector database  → stores and searches vectors (Qdrant)
- Reranker         → improves retrieval precision (cross-encoder)
- LLM              → generates answers from retrieved chunks (Ollama)
