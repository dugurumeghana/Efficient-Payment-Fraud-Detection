# Efficient-Payment-Fraud-Detection
# 💳 Efficient Payment Fraud Detection

This project implements a machine learning solution for identifying fraudulent transactions in financial datasets. Built using Random Forest and deployed via Streamlit, it offers an interactive way to detect suspicious patterns in payment data.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Setup Instructions](#setup-instructions)
- [Demo Screenshots](#demo-screenshots)
- [Folder Structure](#folder-structure)
- [Credits](#credits)
- [Contact](#contact)

---

## 🧠 Overview

Fraudulent activities in the finance sector result in significant losses each year. This project aims to minimize those risks using a supervised machine learning model — **Random Forest** — trained on a well-known dataset of credit card transactions. The solution is designed to help financial institutions or individuals automatically flag suspicious payments with high accuracy.

---

## 🧰 Tech Stack

- **Language**: Python 3.x  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`  
- **ML Algorithm**: Random Forest  
- **Deployment**: Streamlit

---

## ⚙️ How It Works

1. **Data Preprocessing**:
   - Handles missing values
   - Scales features using `StandardScaler`
   - Splits data into training and test sets

2. **Model Training**:
   - A `RandomForestClassifier` is trained on labeled data
   - Evaluated using accuracy, confusion matrix, and ROC-AUC curve

3. **Deployment**:
   - A clean Streamlit interface allows users to upload datasets and view predictions
   - Displays visual summaries of fraud vs. normal transactions

---

## 🚀 Setup Instructions

> ⚠️ The dataset is not included in this repository due to GitHub’s file size restrictions. Please download it from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it inside a `dataset/` folder.

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/dugurumeghana/Efficient-Payment-Fraud-Detection.git

# Navigate into the project
cd Efficient-Payment-Fraud-Detection

# (Optional) Create and activate a virtual environment
python -m venv venv
venv\\Scripts\\activate  # On Windows

# Install the required dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py


Efficient-Payment-Fraud-Detection/
│
├── app.py                     # Streamlit frontend
├── train_fraud_model.ipynb    # Model training notebook
├── rf_model.pkl               # Trained ML model
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── RESULT1.png → RESULT4.png  # Screenshot images for demo
└── dataset/                   # (You add this locally with Kaggle CSV)
