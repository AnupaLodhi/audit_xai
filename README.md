# 🏦 Explainable AI-Based Anomaly Detection for Internal Audit

**Live Demo:** https://auditxai-r6kevqidzvvy2zkutfsnvq.streamlit.app/

## Overview

This academic research prototype helps internal auditors identify unusual banking transactions using unsupervised machine learning. It analyzes uploaded transaction data, calculates anomaly scores, and provides straightforward, rule-based indicators to help reviewers inspect potentially suspicious activity.

> **Note:** An anomaly is not proof of fraud. The current transaction explanations are rule-based indicators, not SHAP or model-derived feature attributions.

## 🎯 Objectives

- Detect unusual transaction behavior without requiring labeled fraud data.
- Assign risk labels to help prioritize transaction reviews.
- Display anomaly scores and understandable transaction indicators.
- Support human auditors in reviewing transaction activity.

## 🛠 Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn (Isolation Forest and StandardScaler)

## ⚙️ Features

- Upload banking transaction data in CSV format.
- Preview uploaded transactions.
- Scale numerical features and run Isolation Forest.
- View anomaly scores and High Risk / Low Risk labels.
- Filter high-risk transactions.
- Select a transaction to see rule-based indicators for high amount, unusual hour, frequency, and location change.
- Download audit results as CSV.

## 🚀 Try the Live Application

**[Open the Streamlit Dashboard](https://auditxai-r6kevqidzvvy2zkutfsnvq.streamlit.app/)**

Download the sample CSV from the app, upload it, and explore the results.

## 📂 Input Format (CSV)

The uploaded CSV must include these columns:

```csv
transaction_id,amount,hour,frequency,location_change
TXN001,15000,14,3,0
TXN002,85000,2,15,1
TXN003,3000,10,2,0
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/AnupaLodhi/audit_xai.git
cd audit_xai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app.py
```

## 📊 Output

- Transaction summary and audit results table
- Anomaly score for each transaction
- High Risk / Low Risk classification
- Rule-based transaction review indicators
- Downloadable results CSV

## 🎓 Academic Use and Limitations

Developed as an academic research prototype exploring machine-learning-assisted internal audit in banking and finance. The current app fits its model on each uploaded dataset and uses a fixed threshold to assign risk labels. It has not been validated as a production fraud detection system or demonstrated regulatory compliance. Future work may include model-faithful explanations such as SHAP, independent evaluation on labeled data, and improved risk calibration.

## 📄 License

See [LICENSE](LICENSE).
