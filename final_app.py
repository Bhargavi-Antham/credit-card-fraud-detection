import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve


OUT_DIR = "models"
ARTIFACT_PATH = os.path.join(OUT_DIR, 'xgboost_final_artifact.pkl')
MAX_DISPLAY = 1000  # max rows to show in Streamlit

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_resource
def load_artifact(path=ARTIFACT_PATH):
    return joblib.load(path)

artifact = load_artifact()
model = artifact['model']
default_threshold = artifact['threshold']
feature_cols = artifact['feature_columns']
scaler_amount = artifact['scaler_amount']
scaler_time = artifact['scaler_time']

st.title("💳 Credit Card Fraud Detection App")
st.write("Upload a CSV file with transactions or enter values manually to predict fraud.")


threshold_input = st.slider(
    "Select threshold for classifying as Fraud",
    min_value=0.0,
    max_value=1.0,
    value=float(default_threshold),
    step=0.01
)

st.subheader("Upload Transaction Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
data = None
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name} successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")


st.subheader("Or Enter Transaction Data Manually")

manual_input = {}
input_features = [f for f in feature_cols if f not in ['Amount_Scaled','Time_Scaled']]

for col in input_features:
    manual_input[col] = st.number_input(f"Enter {col}", value=0.0)


amount_scaled_val = None
if 'Amount_Scaled' in feature_cols:
    amount_choice = st.radio("Amount input type:", ("Raw Amount", "Scaled Amount"))
    if amount_choice == "Raw Amount":
        amount_val = st.number_input("Enter Amount (raw)", value=0.0)
        amount_scaled_val = scaler_amount.transform(pd.DataFrame([[amount_val]], columns=['Amount']))[0][0]
    else:
        amount_scaled_val = st.number_input("Enter Amount_Scaled", value=0.0)
    manual_input['Amount_Scaled'] = amount_scaled_val


time_scaled_val = None
if 'Time_Scaled' in feature_cols:
    time_choice = st.radio("Time input type:", ("Raw Time", "Scaled Time"))
    if time_choice == "Raw Time":
        time_val = st.number_input("Enter Time (raw)", value=0.0)
        time_scaled_val = scaler_time.transform(pd.DataFrame([[time_val]], columns=['Time']))[0][0]
    else:
        time_scaled_val = st.number_input("Enter Time_Scaled", value=0.0)
    manual_input['Time_Scaled'] = time_scaled_val


def predict_transactions(df_new):
    df_proc = df_new.copy()
    
    if 'Amount' in df_proc.columns and 'Amount_Scaled' in feature_cols:
        df_proc['Amount_Scaled'] = scaler_amount.transform(df_proc[['Amount']])
        df_proc.drop(columns=['Amount'], inplace=True)
    if 'Time' in df_proc.columns and 'Time_Scaled' in feature_cols:
        df_proc['Time_Scaled'] = scaler_time.transform(df_proc[['Time']])
        df_proc.drop(columns=['Time'], inplace=True)
    

    for col in feature_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0

    df_proc = df_proc[feature_cols].astype(float)
    probs = model.predict_proba(df_proc)[:,1]
    preds = (probs >= threshold_input).astype(int)
    
    df_new['Prediction'] = preds
    df_new['Probability'] = probs
    return df_new, preds, probs


if st.button("Predict Manual Input") and manual_input:
    df_manual = pd.DataFrame([manual_input])
    try:
        df_manual, preds, probs = predict_transactions(df_manual)
        st.write("### Prediction Result")
        st.write(f"Normal transactions: {(preds==0).sum()}")
        st.write(f"Fraud transactions: {(preds==1).sum()}")
        st.dataframe(df_manual.head(MAX_DISPLAY))

   
        st.write("### Fraud Probability Distribution")
        plt.hist(probs[:10000], bins=50, color='skyblue')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Number of Transactions")
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"Error during prediction: {e}")


if data is not None:
    try:
        df_pred, preds, probs = predict_transactions(data)
        st.write("### Predictions on Uploaded CSV")
        st.write(f"Normal transactions: {(preds==0).sum()}")
        st.write(f"Fraud transactions: {(preds==1).sum()}")

        
        st.dataframe(df_pred.head(MAX_DISPLAY))
        if len(df_pred) > MAX_DISPLAY:
            st.warning(f"File too large, only showing first {MAX_DISPLAY} rows. Download full CSV below.")

        st.write("### Fraud Probability Distribution")
        plt.hist(probs[:10000], bins=50, color='skyblue')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Number of Transactions")
        st.pyplot(plt.gcf())
        plt.clf()

        if 'Class' in data.columns:
            y_true = data['Class']
            st.subheader("Model Evaluation Metrics")
            st.text(classification_report(y_true, preds))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_true, preds))
            st.write(f"ROC-AUC: {roc_auc_score(y_true, probs):.4f}")

        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error during CSV prediction: {e}")
