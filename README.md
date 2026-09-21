# PatternX – AI-Powered Real-Time Cybersecurity Anomaly Detection System

PatternX is a machine learning-based cybersecurity system designed to monitor network traffic and detect suspicious activities.

The project uses the CICIDS2017 dataset and a Random Forest classifier trained on 52 network-flow features to classify network activity as normal traffic or different types of attacks.

## Features

- Network traffic monitoring
- Machine learning-based threat detection
- Detection of DoS, DDoS, Port Scanning, Brute Force, Web Attacks, and Bots
- Risk score and risk level
- Suspicious activity alerts
- Incident investigation
- Host restriction through administrator action
- Security analytics and reports
- Web-based interface using Streamlit

## Technology Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Joblib
- Streamlit
- Plotly

## Dataset

The project uses the **CICIDS2017** network traffic dataset.

The dataset contains normal network traffic and different types of attack activities. The model uses 52 numerical network-flow features for classification.

## Machine Learning Model

A **Random Forest Classifier** is used for network traffic classification.

The model achieved **99.85% test accuracy** on the held-out test dataset used for evaluation.

### Detected Categories

- Normal Traffic
- DoS
- DDoS
- Port Scanning
- Brute Force
- Web Attacks
- Bots

## System Workflow

```text
CICIDS2017 Dataset
        ↓
Data Preprocessing
        ↓
52 Network Features
        ↓
Random Forest Model
        ↓
Detection Result
        ↓
Risk Assessment
        ↓
Alert
        ↓
Investigation
        ↓
Administrator Action
