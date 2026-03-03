import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Internal Audit", layout="wide")

st.title("🏦 Explainable AI for Internal Audit")
st.markdown("Upload a transaction CSV file to detect anomalies and get explainable audit results.")

# Required columns
REQUIRED_COLS = ['transaction_id', 'amount', 'hour', 'frequency', 'location_change']

# Upload CSV
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip().str.lower()

    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Your CSV is missing these required columns: **{', '.join(missing_cols)}**")
        st.info("📋 Your CSV must have these columns: `transaction_id`, `amount`, `hour`, `frequency`, `location_change`")

        st.markdown("### Example CSV format:")
        example = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'amount': [15000, 85000, 3000],
            'hour': [14, 2, 10],
            'frequency': [3, 15, 2],
            'location_change': [0, 1, 0]
        })
        st.dataframe(example)

        # Download sample CSV button
        csv = example.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=csv,
            file_name="sample_transactions.csv",
            mime="text/csv"
        )

    else:
        st.success("✅ File uploaded successfully!")
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
            lambda x: "🔴 High Risk" if x < -0.1 else "🟢 Low Risk"
        )

        # Summary metrics
        st.subheader("📊 Audit Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("High Risk", len(df[df['risk'] == "🔴 High Risk"]))
        col3.metric("Low Risk", len(df[df['risk'] == "🟢 Low Risk"]))

        st.subheader("Audit Results")
        st.dataframe(df[['transaction_id', 'risk', 'anomaly_score']].style.applymap(
            lambda v: 'color: red' if 'High' in str(v) else 'color: green',
            subset=['risk']
        ))

        # Filter high risk only
        if st.checkbox("Show High Risk Transactions Only"):
            st.dataframe(df[df['risk'] == "🔴 High Risk"])

        # Explain individual transaction
        st.subheader("🔍 Explain a Transaction")
        selected = st.selectbox("Select Transaction ID", df['transaction_id'])
        row = df[df['transaction_id'] == selected].iloc[0]

        explanation = []
        if row['amount'] > 40000:
            explanation.append("💰 High transaction amount (above ₹40,000)")
        if row['hour'] < 5:
            explanation.append("🌙 Unusual transaction time (between 12AM–5AM)")
        if row['frequency'] > 10:
            explanation.append("⚡ High transaction frequency (more than 10 times)")
        if row['location_change'] == 1:
            explanation.append("📍 Location change detected")

        st.write("### Explanation")
        if explanation:
            for e in explanation:
                st.warning(e)
        else:
            st.success("✅ Transaction behavior appears normal.")

else:
    st.info("👆 Please upload a CSV file to get started.")

    # Show sample format
    st.markdown("### 📋 Required CSV Format")
    example = pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
        'amount': [15000, 85000, 3000],
        'hour': [14, 2, 10],
        'frequency': [3, 15, 2],
        'location_change': [0, 1, 0]
    })
    st.dataframe(example)

    csv = example.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Sample CSV to Test",
        data=csv,
        file_name="sample_transactions.csv",
        mime="text/csv"
    )
