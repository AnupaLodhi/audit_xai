import numpy as np

def explain_transaction(transaction):
    explanation = []

    if transaction['amount'] > 40000:
        explanation.append("High transaction amount")

    if transaction['hour'] < 5:
        explanation.append("Unusual transaction time")

    if transaction['frequency'] > 10:
        explanation.append("High transaction frequency")

    if transaction['location_change'] == 1:
        explanation.append("Location change detected")

    return explanation
