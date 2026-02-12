🏦 Explainable AI-Based Anomaly Detection for Internal Audit
📌 Overview

This project demonstrates an Explainable Artificial Intelligence (XAI) approach to detect unusual banking transactions for internal audit purposes. The system analyzes transaction data using unsupervised machine learning and provides clear explanations for each flagged transaction, helping auditors understand why a transaction is considered risky.

🎯 Objectives

Detect abnormal transaction behavior without labeled fraud data

Assign risk levels to transactions for audit prioritization

Provide transparent and explainable AI results

Support internal auditors in decision-making

🛠 Technology Stack

Python

Streamlit

Pandas

Scikit-learn

⚙️ Features

Upload transaction data in CSV format

Unsupervised anomaly detection using Isolation Forest

Risk-based transaction classification

Explainable outputs for each flagged transaction

Simple and interactive GUI

📂 Input Format (CSV)

The dataset should contain the following columns:

transaction_id

amount

hour

frequency

location_change

▶️ How to Run the Project

Install required libraries:

pip install streamlit pandas scikit-learn


Run the application:

streamlit run app.py


Open the browser and upload a CSV file to view results.

📊 Output

List of suspicious transactions

Risk level (Low / High)

Anomaly score

Explanation of why a transaction was flagged

🎓 Academic Use

This project is developed as an academic major project and research prototype for demonstrating the application of Explainable AI in banking and finance internal audit systems.