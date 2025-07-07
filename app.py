# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from streamlit.components.v1 import html

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="💳 Efficient Payment Fraud Detection", layout="centered")

# ------------------------------
# Custom Styling with Coolors Palette
# Palette: https://coolors.co/palette/dad7cd-a3b18a-588157-3a5a40-344e41
# ------------------------------
custom_css = """
    <style>
        /* Set background color for main app */
        .main {
            background-color: #A3B18A !important;
            padding: 2rem;
        }

        /* Set sidebar background (optional) */
        .css-1d391kg {
            background-color: #A3B18A !important;
        }

        /* Style buttons */
        .stButton > button {
            background-color: #588157;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
            border: none;
            font-weight: bold;
        }

        /* Style download button */
        .stDownloadButton > button {
            background-color: #3A5A40;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1em;
            border: none;
        }

        /* Metric label color */
        .stMetric label {
            color: #344E41;
        }
    </style>
"""
html(custom_css, height=0)

# ------------------------------
# Title
# ------------------------------
st.markdown("<h1 style='text-align: center; color: #344E41;'>💳 Efficient Payment Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a credit card transactions dataset to identify fraudulent entries.</p>", unsafe_allow_html=True)

# ------------------------------
# Load Model
# ------------------------------
model = joblib.load("rf_model.pkl")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload Credit Card CSV Dataset", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    st.write("### 🧾 Dataset Preview")
    st.dataframe(data.head())

    X = data.drop('Class', axis=1) if 'Class' in data.columns else data

    if st.button("🔍 Predict Fraud Transactions"):
        preds = model.predict(X)
        fraud = int(sum(preds))
        clean = int(len(preds) - fraud)

        data['Prediction'] = preds

        st.write("### 📊 Prediction Results")
        st.dataframe(data.head())

        col1, col2 = st.columns(2)
        col1.metric("✅ Clean Transactions", clean)
        col2.metric("⚠️ Fraud Transactions", fraud)

        fig, ax = plt.subplots()
        ax.bar(["Normal", "Fraud"], [clean, fraud], color=["#344E41", "#D00000"])
        ax.set_title("Transaction Classification Summary")
        st.pyplot(fig)

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime='text/csv'
        )
