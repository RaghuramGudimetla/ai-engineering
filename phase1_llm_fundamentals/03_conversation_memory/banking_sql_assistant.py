import ollama
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

console = Console()

SCHEMA = """
You are an expert Banking SQL assistant with AML and fraud detection expertise.

Database Schema:
- customers(customer_id, name, dob, kyc_status, risk_rating)
- accounts(account_id, customer_id, account_type, balance, status)
- transactions(transaction_id, account_id, amount, type, timestamp, merchant)
- fraud_alerts(alert_id, transaction_id, score, reason, status)
- loans(loan_id, customer_id, amount, interest_rate, status)

Before writing SQL always:
1. UNDERSTAND the business question
2. IDENTIFY the tables and joins needed
3. CONSIDER edge cases (nulls, division by zero, date traps)
4. WRITE clean PostgreSQL with CTEs where needed
5. REVIEW for performance and security

Rules:
- Always use table aliases
- Always add LIMIT unless aggregating
- Use CTEs for complex queries
- Never expose raw PII in results - mask where needed
- Always filter by date ranges on transaction tables
"""

FEW_SHOTS = [
    {
        "role": "user",
        "content": "Show me top 10 customers by total transaction volume this year"
    },
    {
        "role": "assistant",
        "content": """
```sql
WITH customer_volume AS (
    SELECT
        c.customer_id,
        c.name,
        SUM(t.amount) AS total_volume,
        COUNT(t.transaction_id) AS total_transactions
    FROM customers c
    JOIN accounts a ON c.customer_id = a.customer_id
    JOIN transactions t ON a.account_id = t.account_id
    WHERE DATE_PART('year', t.timestamp) = DATE_PART('year', CURRENT_DATE)
    GROUP BY c.customer_id, c.name
)
SELECT *
FROM customer_volume
ORDER BY total_volume DESC
LIMIT 10;
```
"""
    }
]

def extract_sql(text: str) -> str | None:
    """Extract SQL from markdown code blocks"""
    if "```sql" in text:
        start = text.find("```sql") + 6
        end = text.find("```", start)
        return text[start:end].strip()
    return None


def chat(messages: list, question: str) -> str:
    """Send message and get streaming response"""
    messages.append({"role": "user", "content": question})
    
    response = ollama.chat(
        model="llama3.2",
        messages=messages,
        stream=True
    )
    
    full_response = ""
    for chunk in response:
        full_response += chunk.message.content
    
    messages.append({"role": "assistant", "content": full_response})
    return full_response


def display_response(response: str):
    """Pretty print the response"""
    sql = extract_sql(response)
    
    if sql:
        # Display any text before the SQL
        pre_sql = response[:response.find("```sql")].strip()
        if pre_sql:
            console.print(Markdown(pre_sql))
        
        # Display SQL with syntax highlighting
        syntax = Syntax(sql, "sql", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated SQL", border_style="green"))
    else:
        console.print(Markdown(response))


def main():
    console.print(Panel.fit(
        "[bold green]🏦 Banking SQL Assistant[/bold green]\n"
        "[dim]Powered by Llama3.2 + Ollama[/dim]\n\n"
        "[yellow]Commands:[/yellow]\n"
        "  [cyan]exit[/cyan]     - Quit\n"
        "  [cyan]clear[/cyan]    - Clear conversation history\n"
        "  [cyan]history[/cyan]  - Show conversation history",
        border_style="green"
    ))

    # Build initial messages with system prompt and few shots
    messages = [{"role": "system", "content": SCHEMA}]
    messages += FEW_SHOTS

    while True:
        try:
            # Get user input
            console.print("\n[bold cyan]You:[/bold cyan] ", end="")
            question = input().strip()

            if not question:
                continue

            if question.lower() == "exit":
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break

            if question.lower() == "clear":
                messages = [{"role": "system", "content": SCHEMA}]
                messages += FEW_SHOTS
                console.print("[yellow]Conversation cleared.[/yellow]")
                continue

            if question.lower() == "history":
                for i, msg in enumerate(messages):
                    if msg["role"] != "system":
                        console.print(f"\n[dim]{msg['role'].upper()}:[/dim] {msg['content'][:100]}...")
                continue

            # Get and display response
            console.print("\n[bold green]Assistant:[/bold green]")
            response = chat(messages, question)
            display_response(response)

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye! 👋[/yellow]")
            break


if __name__ == "__main__":
    main()
