# RAG - Retrieval Augmented Generation

## The Problem RAG Solves
LLMs are trained on data up to a cutoff date and have no knowledge of your
internal documents, policies or private data. You cannot solve this by putting
everything in the system prompt because context windows are limited.
RAG solves this cleanly.

## What is RAG?
RAG gives LLMs access to your own data at query time by retrieving only the
relevant pieces and injecting them into the prompt.

## How it Works

### Indexing (done once)
1. Load your documents
2. Split into chunks
3. Convert each chunk into a vector (embedding)
4. Store vectors in a vector database

### Querying (done every time)
1. Convert user question into a vector
2. Search vector database for similar vectors
3. Retrieve top K most relevant chunks
4. Pass question + chunks to LLM
5. LLM generates answer using only retrieved chunks

## Chunking Strategies

### Fixed Size
Split by character count regardless of content structure.
Simple but often cuts sentences in half.
Use for: Unstructured or mixed content documents.

### Paragraph
Split by double newlines - natural paragraph breaks.
Respects content boundaries.
Use for: General text like articles, emails, notes.

### Section
Split by document sections and headings.
Keeps related content together.
Use for: Structured documents like policies, runbooks, legal docs.

### The Rule of Thumb
- Structured docs (policies, legal, runbooks) → Section chunking
- General text (articles, emails, notes)      → Paragraph chunking
- Unstructured or mixed content               → Fixed size with overlap
- Code files                                  → Function or class level
- Tables and CSVs                             → Row or row-group chunking

## Why Chunking Matters
Bad chunking = bad retrieval = bad answers regardless of how good your LLM is.
Chunking strategy is the most underrated but critical decision in RAG.

## The Payload
Every chunk stored in the vector database should carry metadata alongside the text.
- text    → the actual content, used to generate the answer
- source  → which document it came from, used for citations
- section → which part of the document, used for precise citations

In banking a compliance answer without a citation is worthless.
The payload is what makes answers traceable and auditable.
