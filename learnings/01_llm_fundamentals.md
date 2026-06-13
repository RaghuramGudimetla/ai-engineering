# LLM Fundamentals

## What is an LLM?
A Large Language Model is a neural network trained on massive amounts of text.
It learns patterns in language and can generate human-like text responses.
Examples: Claude, GPT-4, Llama, Mistral.

## Key Concepts

### Tokens
LLMs read tokens not words. A token is roughly 4 characters or 0.75 words.
Token count matters because every model has a maximum token limit per request.

### Context Window
Maximum amount of text an LLM can read in one request including system prompt,
conversation history, retrieved chunks and the response.

### Temperature
Controls how creative or random the responses are.
- 0.0 = deterministic, best for SQL and code
- 0.5 = balanced, good for most use cases
- 1.0 = creative, good for writing

### System Prompt
Instructions given to the LLM before the conversation starts.
Defines persona, rules, constraints and context.
Think of it as the job description you give an employee before they start.

### LLM Memory
LLMs are stateless. They remember nothing between API calls.
What feels like memory is just passing the full conversation history every request.

### Streaming
Instead of waiting for the full response, streaming prints tokens as they are
generated like ChatGPT typing in real time. Every production app uses streaming.

## Ollama
Runs LLMs locally on your machine. No API key, no cost, no data leaving your computer.
- ollama pull llama3.2  -> download a model
- ollama serve          -> start the local server
- ollama list           -> see installed models
