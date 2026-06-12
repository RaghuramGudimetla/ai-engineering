import ollama

SCHEMA = """
You are an expert Banking SQL assistant. Generate only valid PostgreSQL queries.

Database Schema:
- customers(customer_id, name, dob, kyc_status, risk_rating)
- accounts(account_id, customer_id, account_type, balance, status)
- transactions(transaction_id, account_id, amount, type, timestamp, merchant)
- fraud_alerts(alert_id, transaction_id, score, reason, status)
- loans(loan_id, customer_id, amount, interest_rate, status)

Rules:
- Always use table aliases
- Always add a LIMIT clause unless aggregating
- Use CTEs for complex queries
- Return ONLY the SQL query, no explanation
"""

# Few-shot examples — teaching the LLM your style
FEW_SHOTS = [
    {
        "role": "user",
        "content": "Show me top 10 customers by total transaction volume this year"
    },
    {
        "role": "assistant",
        "content": """
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
        """
    },
    {
        "role": "user",
        "content": "Which accounts have more than 3 fraud alerts this month?"
    },
    {
        "role": "assistant",
        "content": """
WITH monthly_alerts AS (
    SELECT
        t.account_id,
        COUNT(fa.alert_id) AS alert_count
    FROM transactions t
    JOIN fraud_alerts fa ON t.transaction_id = fa.transaction_id
    WHERE DATE_TRUNC('month', t.timestamp) = DATE_TRUNC('month', CURRENT_DATE)
    GROUP BY t.account_id
    HAVING COUNT(fa.alert_id) > 3
)
SELECT
    a.account_id,
    a.account_type,
    c.name,
    ma.alert_count
FROM monthly_alerts ma
JOIN accounts a ON ma.account_id = a.account_id
JOIN customers c ON a.customer_id = c.customer_id
ORDER BY ma.alert_count DESC
LIMIT 50;
        """
    }
]

# Now ask real questions — LLM follows the style from examples above
questions = [
    "Find customers with high risk rating who have made transactions over $50,000",
    "Show me all open loans where the customer has an active fraud alert",
    "Which merchants have the highest number of flagged transactions this quarter?"
]

for question in questions:
    print(f"\n Question: {question}")
    print("-" * 60)

    messages = [{"role": "system", "content": SCHEMA}]
    messages += FEW_SHOTS  # inject examples
    messages += [{"role": "user", "content": question}]

    response = ollama.chat(
        model="llama3.2",
        messages=messages
    )

    print(response.message.content)
