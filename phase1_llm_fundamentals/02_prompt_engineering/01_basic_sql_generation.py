import ollama

SCHEMA = """
You are an expert Banking SQL assistant. Generate only valid Snowflake queries.

Database Schema:
- customers(customer_id, name, dob, kyc_status, risk_rating)
- accounts(account_id, customer_id, account_type, balance, status)
- transactions(transaction_id, account_id, amount, type, timestamp, merchant)
- fraud_alerts(alert_id, transaction_id, score, reason, status)
- loans(loan_id, customer_id, amount, interest_rate, status)

Rules:
- Always use table aliases
- Always add a LIMIT clause unless aggregating
- Return ONLY the SQL query, no explanation
"""

questions = [
    "Show me all transactions over $10,000 in the last 30 days",
    "Which customers have a high risk rating and pending KYC status?",
    "Find all fraud alerts with a score above 0.8 that are still open",
]

for question in questions:
    print(f"\n Question: {question}")
    print("-" * 60)

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SCHEMA},
            {"role": "user", "content": question}
        ]
    )

    print(response.message.content)
