import ollama

SCHEMA = """
Database Schema:
- customers(customer_id, name, dob, kyc_status, risk_rating)
- accounts(account_id, customer_id, account_type, balance, status)
- transactions(transaction_id, account_id, amount, type, timestamp, merchant)
- fraud_alerts(alert_id, transaction_id, score, reason, status)
- loans(loan_id, customer_id, amount, interest_rate, status)
"""

GENERATOR_PROMPT = f"""
You are a Banking SQL developer. Write PostgreSQL queries based on the request.
{SCHEMA}
Return ONLY the SQL query, no explanation.
"""

CRITIC_PROMPT = f"""
You are a Senior Banking SQL reviewer with 20 years experience.
You review SQL queries for:

1. CORRECTNESS   - Wrong joins, missing filters, incorrect aggregations
2. PERFORMANCE   - Missing indexes hints, unnecessary full table scans
3. SECURITY      - SQL injection risks, exposure of sensitive data (PII)
4. BANKING RULES - Missing date filters, no LIMIT on large tables,
                   unhandled NULL values in amount fields
5. EDGE CASES    - Division by zero, empty result handling

{SCHEMA}

Structure your review exactly like this:

## Issues Found
- [CRITICAL/WARNING/INFO] Issue description

## Verdict
APPROVED or NEEDS REVISION

## Revised SQL (only if NEEDS REVISION)
```sql
-- improved query
```
"""

def generate_sql(question: str) -> str:
    """Step 1: Generate initial SQL"""
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": GENERATOR_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    return response.message.content


def critique_sql(question: str, sql: str) -> str:
    """Step 2: Critique and improve the SQL"""
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"""
Original Question: {question}

SQL to Review:
{sql}

Please review this SQL thoroughly.
"""}
        ],
        stream=True
    )
    
    result = ""
    for chunk in response:
        print(chunk.message.content, end="", flush=True)
        result += chunk.message.content
    return result


# Intentionally tricky questions that will generate imperfect SQL
questions = [
    "Calculate the fraud rate percentage per account type",
    "Find customers whose total transaction amount doubled compared to last month",
    "Show average loan amount by risk rating including customers with no loans"
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"📝 Question: {question}")
    print('='*60)

    # Step 1 - Generate
    print("\n⚙️  Generating SQL...")
    initial_sql = generate_sql(question)
    print("\n📄 Initial SQL:")
    print(initial_sql)

    # Step 2 - Critique
    print("\n🔍 Reviewing SQL...\n")
    critique_sql(question, initial_sql)
    print("\n")
