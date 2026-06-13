import os

documents_path = "phase2_rag/documents"

def load_document(filename: str) -> str:
    with open(f"{documents_path}/{filename}", "r") as f:
        return f.read()

# ------------------------------------------------------------
# STRATEGY 1 - Fixed Size Chunking
# Split by character count regardless of content structure.
# Simple but dumb - often cuts sentences in half.
# ------------------------------------------------------------

def fixed_size_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ------------------------------------------------------------
# STRATEGY 2 - Paragraph Chunking
# Split by double newlines (natural paragraph breaks).
# Better than fixed size - respects content boundaries.
# ------------------------------------------------------------

def paragraph_chunks(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

# ------------------------------------------------------------
# STRATEGY 3 - Section Chunking
# Split by document sections (headings).
# Best for structured documents like policies and runbooks.
# ------------------------------------------------------------

def section_chunks(text: str) -> list[str]:
    lines = text.split("\n")
    chunks = []
    current_chunk = []

    for line in lines:
        is_header = (
            line.isupper() and len(line) > 5 or
            line.startswith("SECTION") or
            line.startswith("CHAPTER")
        )

        if is_header and current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = [line]
        else:
            current_chunk.append(line)

    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks

def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def analyse_chunks(chunks: list[str], strategy_name: str):
    print_separator(strategy_name)
    print(f"Total chunks   : {len(chunks)}")
    print(f"Avg chunk size : {sum(len(c) for c in chunks) // len(chunks)} chars")
    print(f"Smallest chunk : {min(len(c) for c in chunks)} chars")
    print(f"Largest chunk  : {max(len(c) for c in chunks)} chars")
    print(f"\nFirst chunk preview:")
    print("-" * 40)
    print(chunks[0][:300])
    print(f"\nSecond chunk preview:")
    print("-" * 40)
    print(chunks[1][:300])

text = load_document("aml_policy.txt")
print(f"Document loaded: {len(text)} characters")

fixed  = fixed_size_chunks(text, chunk_size=200, overlap=50)
para   = paragraph_chunks(text)
sec    = section_chunks(text)

analyse_chunks(fixed, "Strategy 1 - Fixed Size (200 chars, 50 overlap)")
analyse_chunks(para,  "Strategy 2 - Paragraph Chunking")
analyse_chunks(sec,   "Strategy 3 - Section Chunking")

print_separator("Key Insight")
print("""
Fixed Size  - Fast but cuts sentences mid-way. Use for unstructured docs.
Paragraph   - Respects natural breaks. Good general purpose strategy.
Section     - Best for structured policy/legal documents.
              Keeps related content together which improves retrieval.

For our banking documents we will use Section chunking.
""")
