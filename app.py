
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.markdown("---")

@st.cache_data
def load_data():
    data = pd.read_csv("creditcard_sample.csv")
    return data

@st.cache_resource
def train_model(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    scaler = StandardScaler()
    X['Time'] = scaler.fit_transform(X[['Time']])
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    model = ExtraTreesClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X_train_smote, y_train_smote)
    return model, X_test, y_test, X

# Load data
with st.spinner("Loading dataset..."):
    data = load_data()

st.success(f"✅ Dataset loaded: {data.shape[0]:,} rows, {data.shape[1]} columns")

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", [
    "📊 Data Overview",
    "🔍 Class Distribution",
    "🤖 Train & Evaluate Model",
    "📈 Feature Importance",
    "🔮 Predict a Transaction"
])

# ── Data Overview ──────────────────────────────────────────
if section == "📊 Data Overview":
    st.header("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(data):,}")
    col2.metric("Fraudulent", f"{data['Class'].sum():,}")
    col3.metric("Fraud %", f"{data['Class'].mean()*100:.4f}%")

    st.subheader("Sample Data")
    st.dataframe(data.head(10))

    st.subheader("Missing Values")
    nulls = data.isnull().sum()
    st.write("Total nulls:", int(nulls.sum()))

# ── Class Distribution ─────────────────────────────────────
elif section == "🔍 Class Distribution":
    st.header("🔍 Class Distribution")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Class', data=data, ax=ax, palette=['steelblue', 'crimson'])
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_title("Transaction Class Distribution")
    st.pyplot(fig)
    st.write(data['Class'].value_counts().rename({0: 'Legitimate', 1: 'Fraud'}))

# ── Train & Evaluate ───────────────────────────────────────
elif section == "🤖 Train & Evaluate Model":
    st.header("🤖 Model Training & Evaluation")
    with st.spinner("Training ExtraTreesClassifier with SMOTE... (may take a minute)"):
        model, X_test, y_test, X = train_model(data)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    col3.metric("Test Samples", f"{len(y_test):,}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc_score(y_test, y_prob):.4f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

# ── Feature Importance ─────────────────────────────────────
elif section == "📈 Feature Importance":
    st.header("📈 Feature Importance")
    with st.spinner("Training model..."):
        model, X_test, y_test, X = train_model(data)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax, palette='viridis')
    ax.set_title("Top 15 Feature Importances")
    st.pyplot(fig)
    st.dataframe(importance.reset_index(drop=True))

# ── Predict a Transaction ──────────────────────────────────
elif section == "🔮 Predict a Transaction":
    st.header("🔮 Predict a Single Transaction")
    with st.spinner("Loading model..."):
        model, X_test, y_test, X = train_model(data)

    st.info("Pick a row index from the test set to predict.")
    idx = st.slider("Test set row index", 0, len(X_test) - 1, 0)
    sample = X_test.iloc[idx].values.reshape(1, -1)

    if st.button("🔍 Predict"):
        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]
        if pred == 0:
            st.success(f"✅ **Legitimate Transaction** (Fraud probability: {prob*100:.2f}%)")
        else:
            st.error(f"🚨 **Fraudulent Transaction** (Fraud probability: {prob*100:.2f}%)")
        st.write("**Actual label:**", "Fraud" if y_test.iloc[idx] == 1 else "Legitimate")
