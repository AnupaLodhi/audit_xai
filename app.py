import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Internal Audit", layout="wide")

st.title("🏦 Explainable AI for Internal Audit")

# Upload CSV
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Transaction Data Preview")
    st.dataframe(df)

    # Select features
    features = df[['amount', 'hour', 'frequency', 'location_change']]

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Train model
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(X)

    # Predict anomaly score
    df['anomaly_score'] = model.decision_function(X)

    # Risk classification
    df['risk'] = df['anomaly_score'].apply(
        lambda x: "High Risk" if x < -0.1 else "Low Risk"
    )

    st.subheader("Audit Results")
    st.dataframe(df[['transaction_id', 'risk', 'anomaly_score']])

    st.subheader("Explain Transaction")
    selected = st.selectbox("Select Transaction ID", df['transaction_id'])

    row = df[df['transaction_id'] == selected].iloc[0]

    explanation = []
    if row['amount'] > 40000:
        explanation.append("High transaction amount")
    if row['hour'] < 5:
        explanation.append("Unusual transaction time")
    if row['frequency'] > 10:
        explanation.append("High transaction frequency")
    if row['location_change'] == 1:
        explanation.append("Location change detected")

    st.write("### Explanation")
    if explanation:
        for e in explanation:
            st.write("•", e)
    else:
        st.write("Transaction behavior is normal")
