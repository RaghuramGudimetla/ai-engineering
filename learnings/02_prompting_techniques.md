# Prompting Techniques

Prompting techniques are design patterns for getting the best out of LLMs.
The same model with different prompts produces dramatically different results.

## Zero-Shot
Just ask the question with no examples. LLM uses its training knowledge.
Use when: Simple clear tasks.

## Few-Shot
Show examples of what you want before asking the real question.
Teaching by example rather than instruction.
Use when: You need output in a specific style or format.

## Chain of Thought (CoT)
Force the LLM to think step by step before answering.
Reasoning out loud improves quality because each reasoning token becomes
context for the next token.
Use when: Complex multi-step reasoning, multi-table SQL, AML logic.

## ReAct (Reason + Act)
LLM reasons about what to do, takes an action, observes the result,
then reasons again. Loop continues until it has a final answer.
This is the foundation of AI agents.
Use when: Multi-step investigations where you do not know upfront what data you need.

## Self-Critique
LLM generates an answer, reviews its own output for errors, then rewrites
an improved version. Three steps: Generate, Critique, Revise.
Use when: High stakes outputs like SQL on production databases.

## Role Prompting
Give the LLM a specific persona and expertise level.
Use when: You need domain specific expertise and tone.

## Prompt Chaining
Output of one prompt becomes input of the next.
Break complex tasks into a pipeline of simpler prompts.
Use when: Tasks too complex for a single prompt.

## Quick Reference
- Zero-shot      → simple clear tasks
- Few-shot       → specific style needed
- Chain of Thought → complex reasoning
- ReAct          → multi-step tool use and agents
- Self-critique  → high stakes outputs
- Role prompting → domain expertise needed
- Prompt chaining → complex multi-stage tasks
