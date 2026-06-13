# Phase 2 - RAG (Retrieval Augmented Generation)

## The Problem

Large Language Models are trained on data up to a certain date.
They have no knowledge of your internal documents, policies, or private data.

You cannot solve this by:
- Putting everything in the system prompt - context windows are limited
- Retraining the model - extremely expensive and slow
- Fine-tuning - good for style/behaviour, not for injecting knowledge

RAG solves this cleanly.

## What is RAG?

RAG is a technique that gives LLMs access to your own data at query time
by retrieving only the relevant pieces and injecting them into the prompt.

## How it Works

### Step 1 - Indexing (done once)

Your Documents
     |
     v
Split into chunks (paragraphs, sections)
     |
     v
Convert each chunk into a vector (embedding)
     |
     v
Store vectors in a Vector Database

### Step 2 - Querying (done every time)

User Question
     |
     v
Convert question into a vector (same embedding model)
     |
     v
Search Vector DB for similar vectors
     |
     v
Retrieve top K most relevant chunks
     |
     v
LLM(question + retrieved chunks) = Answer

## Why Vectors?

A vector is a list of numbers that represents the meaning of text.
Similar meaning = similar vectors = close together in vector space.

Example:
- "customer due diligence" and "KYC process" are semantically similar
- Their vectors will be close together
- So searching for one will find the other

This is called semantic search - searching by meaning not keywords.

## Why RAG Matters in Banking

Banks have thousands of pages of:
- AML and KYC policies
- Regulatory guidelines (Basel III, FATF)
- Fraud investigation runbooks
- Product manuals and compliance rules

Without RAG: Analysts manually search documents to answer questions.
With RAG: Ask a question in plain English and get an accurate answer
          with references back to the source document.

## What We Build in This Phase

A Banking Policy Q&A system that:
- Ingests internal banking policy documents
- Answers compliance and AML questions in plain English
- Cites exactly which document and section the answer came from
- Never makes up answers - only uses your documents

## Key Concepts Covered

- Chunking strategies - how to split documents intelligently
- Embedding models - converting text to vectors
- Vector databases - storing and searching vectors (Qdrant)
- Retrieval - finding the right chunks for a question
- Generation - using retrieved chunks to answer accurately
- Evaluation - measuring how good your RAG system is

## Stack

- sentence-transformers - embedding model (runs locally)
- Qdrant - vector database (runs locally via Docker)
- Ollama - LLM (runs locally)
- pypdf - reading PDF documents
