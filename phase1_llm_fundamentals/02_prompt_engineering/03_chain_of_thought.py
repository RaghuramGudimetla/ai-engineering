import ollama

SCHEMA = """
You are an expert Banking SQL assistant. Generate only valid PostgreSQL queries.

Database Schema:
- customers(customer_id, name, dob, kyc_status, risk_rating)
- accounts(account_id, customer_id, account_type, balance, status)
- transactions(transaction_id, account_id, amount, type, timestamp, merchant)
- fraud_alerts(alert_id, transaction_id, score, reason, status)
- loans(loan_id, customer_id, amount, interest_rate, status)

Before writing ANY SQL, you must think through:
1. UNDERSTAND - What is the business question being asked?
2. TABLES - Which tables are needed and why?
3. JOINS - How do the tables connect?
4. FILTERS - What WHERE conditions are needed?
5. AGGREGATIONS - Any GROUP BY, HAVING, window functions?
6. EDGE CASES - Any nulls, duplicates, or date traps to handle?
7. SQL - Now write the final clean PostgreSQL query

Always structure your response exactly like this:

## Thinking
1. UNDERSTAND: ...
2. TABLES: ...
3. JOINS: ...
4. FILTERS: ...
5. AGGREGATIONS: ...
6. EDGE CASES: ...

## SQL
```sql
-- your query here
```
"""

# Complex banking questions that NEED reasoning to get right
questions = [
    """
    Find customers who:
    - Have more than 2 accounts
    - At least one account has a fraud alert with score > 0.85
    - Have an active loan
    - KYC status is still pending
    This is for an urgent AML review.
    """,
    """
    Show me a monthly trend of suspicious transactions for the last 6 months.
    A transaction is suspicious if it has a fraud alert OR 
    the amount is more than 3x the customer's average transaction amount.
    """,
    """
    Find all customers who have made transactions to the same merchant 
    as another customer who has an open fraud alert.
    This helps identify potential fraud rings.
    """
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question.strip()}")
    print('='*60)

    stream = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SCHEMA},
            {"role": "user", "content": question}
        ],
        stream=True
    )

    for chunk in stream:
        print(chunk.message.content, end="", flush=True)

    print("\n")
