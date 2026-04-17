import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Internal Audit", layout="wide")

st.title("🏦 Explainable AI for Internal Audit")
st.markdown("Upload a transaction CSV file to detect anomalies and get explainable audit results.")

# Required columns
REQUIRED_COLS = ['transaction_id', 'amount', 'hour', 'frequency', 'location_change']

# ── Helper: color risk column safely (works on ALL pandas versions) ──
def color_risk_col(df_display):
    def highlight(val):
        if 'High' in str(val):
            return 'color: red; font-weight: bold'
        return 'color: green; font-weight: bold'
    try:
        return df_display.style.map(highlight, subset=['risk'])
    except AttributeError:
        return df_display.style.applymap(highlight, subset=['risk'])

# ── Upload CSV ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Your CSV is missing these required columns: **{', '.join(missing_cols)}**")
        st.info("📋 Your CSV must have these columns: `transaction_id`, `amount`, `hour`, `frequency`, `location_change`")

        st.markdown("### Example CSV format:")
        example = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'amount':         [15000,    85000,    3000],
            'hour':           [14,       2,        10],
            'frequency':      [3,        15,       2],
            'location_change':[0,        1,        0]
        })
        st.dataframe(example)
        csv = example.to_csv(index=False)
        st.download_button("⬇️ Download Sample CSV", csv, "sample_transactions.csv", "text/csv")

    else:
        st.success("✅ File uploaded successfully!")
        st.subheader("Transaction Data Preview")
        st.dataframe(df)

        # Feature scaling
        features = df[['amount', 'hour', 'frequency', 'location_change']]
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        # Isolation Forest
        model = IsolationForest(contamination=0.2, random_state=42)
        model.fit(X)
        df['anomaly_score'] = model.decision_function(X).round(4)
        df['risk'] = df['anomaly_score'].apply(
            lambda x: "🔴 High Risk" if x < -0.1 else "🟢 Low Risk"
        )

        # Summary metrics
        st.subheader("📊 Audit Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("🔴 High Risk", len(df[df['risk'] == "🔴 High Risk"]))
        col3.metric("🟢 Low Risk",  len(df[df['risk'] == "🟢 Low Risk"]))

        # Audit Results Table
        st.subheader("📋 Audit Results")
        display_df = df[['transaction_id', 'risk', 'anomaly_score']].copy()
        st.dataframe(color_risk_col(display_df), use_container_width=True)

        # Download results
        result_csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Full Audit Results", result_csv, "audit_results.csv", "text/csv")

        # Filter high risk
        if st.checkbox("🔍 Show High Risk Transactions Only"):
            st.dataframe(df[df['risk'] == "🔴 High Risk"], use_container_width=True)

        # Explain a transaction
        st.subheader("🔍 Explain a Transaction")
        selected = st.selectbox("Select Transaction ID", df['transaction_id'].tolist())
        row = df[df['transaction_id'] == selected].iloc[0]

        st.markdown(f"**Risk Level:** {row['risk']}  |  **Anomaly Score:** `{row['anomaly_score']}`")

        explanation = []
        if row['amount'] > 40000:
            explanation.append("💰 High transaction amount (above ₹40,000)")
        if row['hour'] < 5:
            explanation.append("🌙 Unusual transaction time (between 12AM – 5AM)")
        if row['frequency'] > 10:
            explanation.append("⚡ High transaction frequency (more than 10 times)")
        if row['location_change'] == 1:
            explanation.append("📍 Location change detected")

        st.write("### 📝 Explanation")
        if explanation:
            for e in explanation:
                st.warning(e)
        else:
            st.success("✅ Transaction behavior appears normal.")

else:
    st.info("👆 Please upload a CSV file to get started.")
    st.markdown("### 📋 Required CSV Format")
    example = pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
        'amount':         [15000,    85000,    3000],
        'hour':           [14,       2,        10],
        'frequency':      [3,        15,       2],
        'location_change':[0,        1,        0]
    })
    st.dataframe(example)
    csv = example.to_csv(index=False)
    st.download_button("⬇️ Download Sample CSV to Test", csv, "sample_transactions.csv", "text/csv")
