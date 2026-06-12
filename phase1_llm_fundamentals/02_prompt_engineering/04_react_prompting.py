import ollama
import json

# Simulated database results - in real life these would be actual DB calls
def execute_sql(sql: str) -> dict:
    """Simulates executing SQL and returning results"""
    
    # Simulate different results based on keywords in the SQL
    if "fraud_alerts" in sql.lower() and "score" in sql.lower():
        return {
            "rows": [
                {"account_id": "ACC001", "customer_name": "John Smith", "alert_score": 0.92, "amount": 45000},
                {"account_id": "ACC007", "customer_name": "Sarah Jones", "alert_score": 0.88, "amount": 78000},
                {"account_id": "ACC012", "customer_name": "Mike Brown", "alert_score": 0.95, "amount": 120000},
            ],
            "row_count": 3
        }
    elif "loans" in sql.lower():
        return {
            "rows": [
                {"customer_id": "C001", "customer_name": "John Smith", "loan_amount": 250000, "status": "active"},
                {"customer_id": "C012", "customer_name": "Mike Brown", "loan_amount": 500000, "status": "active"},
            ],
            "row_count": 2
        }
    elif "transactions" in sql.lower() and "merchant" in sql.lower():
        return {
            "rows": [
                {"merchant": "FastCash Ltd", "transaction_count": 47, "total_amount": 380000},
                {"merchant": "QuickPay Inc", "transaction_count": 31, "total_amount": 220000},
            ],
            "row_count": 2
        }
    else:
        return {"rows": [], "row_count": 0}


SYSTEM_PROMPT = """
You are an expert Banking AML (Anti Money Laundering) SQL analyst.

Database Schema:
- customers(customer_id, name, dob, kyc_status, risk_rating)
- accounts(account_id, customer_id, account_type, balance, status)
- transactions(transaction_id, account_id, amount, type, timestamp, merchant)
- fraud_alerts(alert_id, transaction_id, score, reason, status)
- loans(loan_id, customer_id, amount, interest_rate, status)

You work in a loop:
1. THOUGHT: Reason about what you need to investigate
2. ACTION: Write a SQL query to get that data
3. OBSERVATION: Analyse the results returned
4. REPEAT or FINAL ANSWER: Either investigate further or summarise findings

Always format your response exactly like this:

THOUGHT: [your reasoning]
ACTION:
```sql
[your SQL query]
```

When you have enough information:
FINAL ANSWER: [your complete AML investigation summary]
"""

def run_react_agent(investigation_request: str):
    print(f"\n{'='*60}")
    print(f"AML INVESTIGATION REQUEST:")
    print(f"{investigation_request}")
    print('='*60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": investigation_request}
    ]

    # ReAct loop - max 4 iterations
    for step in range(1, 5):
        print(f"\n--- Step {step} ---")

        response = ollama.chat(
            model="llama3.2",
            messages=messages
        )

        assistant_message = response.message.content
        print(assistant_message)

        # If LLM has reached final answer, stop
        if "FINAL ANSWER" in assistant_message:
            break

        # Extract SQL from the response
        if "```sql" in assistant_message:
            sql_start = assistant_message.find("```sql") + 6
            sql_end = assistant_message.find("```", sql_start)
            sql = assistant_message[sql_start:sql_end].strip()

            # Execute the SQL (simulated)
            print(f"\n⚙️  Executing SQL...")
            result = execute_sql(sql)
            print(f"📊 Result: {result['row_count']} rows returned")

            # Feed result back to LLM as observation
            observation = f"""
OBSERVATION: SQL executed successfully.
Results ({result['row_count']} rows):
{json.dumps(result['rows'], indent=2)}

Continue your investigation or provide FINAL ANSWER if you have enough information.
"""
            # Add assistant response and observation to message history
            messages.append({"role": "assistant", "content": assistant_message})
            messages.append({"role": "user", "content": observation})

        else:
            # No SQL found, LLM is done
            break


# Run an AML investigation
run_react_agent("""
Investigate potential money laundering activity. 
Start by identifying high risk fraud alerts, 
then check if those customers have active loans,
then look for suspicious merchant patterns.
Provide a final AML risk summary.
""")

