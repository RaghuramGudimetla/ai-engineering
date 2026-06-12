import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "What is RAG in AI? Explain in 3 bullet points."}
    ]
)

print(response.message.content)
