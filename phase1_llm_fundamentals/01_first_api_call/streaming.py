import ollama

stream = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "What is RAG in AI? Explain in 3 bullet points."}
    ],
    stream=True
)

for chunk in stream:
    print(chunk.message.content, end="", flush=True)

print()  # newline at the end
