# 💳 Credit Card Fraud Detection using SMOTE and ExtraTreesClassifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srijakothakonda/creditcard_Fraud-detection/blob/main/credit_card_fraud_detection.ipynb)

**Live Demo:** [Streamlit Web App](https://creditcardfraud-detection-5wkfuzithcuygwydnkjvin.streamlit.app/)

---

## 📌 Project Overview

Credit card fraud causes major financial losses, but fraudulent transactions are extremely rare. In this dataset only **0.17%** of transactions are fraud, which makes the data **highly imbalanced**.

This project builds a machine learning model that detects fraudulent transactions. **SMOTE** (Synthetic Minority Over-sampling Technique) balances the training data, and an **ExtraTreesClassifier** is trained to classify each transaction as legitimate or fraudulent. The model is also deployed as an interactive **Streamlit** web application.

---

## 📊 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 columns
- **Fraud cases:** 492 (0.1727%)
- **Missing values:** None

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction |
| `V1 – V28` | Anonymized features obtained using PCA |
| `Amount` | Transaction amount |
| `Class` | Target: `0` = Legitimate, `1` = Fraud |

> The full dataset is not stored in this repository because of its size. `creditcard_sample.csv` (50,000 rows) is included for the Streamlit demo.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Joblib
- Streamlit

---

## 🔄 Project Workflow

1. **Load dataset** and check shape, missing values, and duplicates
2. **Exploratory analysis:** class distribution, Amount and Time behaviour
3. **Train-test split:** 80/20 with `stratify=y` to keep the fraud ratio in both sets
4. **Feature scaling:** `StandardScaler` applied to `Time` and `Amount`, fitted on **training data only** to avoid data leakage
5. **SMOTE:** applied on the **training set only**; the test set stays untouched
6. **Model training:** `ExtraTreesClassifier`
7. **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, PR-AUC
8. **Feature importance** and threshold analysis
9. **Save model** with Joblib
10. **Deploy** using Streamlit

---

## 🤖 Machine Learning Model

**Algorithm:** ExtraTreesClassifier (Extremely Randomized Trees)

| Parameter | Value |
|---|---|
| `n_estimators` | 20 |
| `random_state` | 42 |

**Why ExtraTrees?**
- An ensemble of decision trees that reduces variance and overfitting
- Fast to train and works well on tabular data
- Handles non-linear patterns and provides feature importance
- Light enough to run on free Streamlit hosting

**Why SMOTE?**
SMOTE creates new synthetic fraud samples by interpolating between a fraud sample and its nearest fraud neighbours. In this project it balanced the training data from **394 → 227,451** fraud samples, matching the legitimate class.

---

## 📈 Results (Full Dataset – 284,807 Transactions)

Evaluated on an untouched test set of 56,962 transactions (98 fraud cases).

| Metric | Score |
|---|---|
| Accuracy | **99.95%** |
| Precision (fraud) | **0.87** |
| Recall (fraud) | **0.84** |
| F1-Score (fraud) | **0.85** |
| ROC-AUC | **0.956** |
| PR-AUC | **0.846** |

### Confusion Matrix

|  | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | 56,852 (TN) | 12 (FP) |
| **Actual Fraud** | 16 (FN) | 82 (TP) |

- Out of 98 real frauds, the model caught **82** and missed **16**.
- Out of 56,864 legitimate transactions, only **12** were wrongly flagged.

> **Note on accuracy:** because 99.83% of transactions are legitimate, accuracy alone is misleading. A model that predicts "legitimate" every time would also score about 99.8%. Precision, recall, F1, and PR-AUC on the fraud class give the true picture.

---

## 🌐 Streamlit Web Application

The app (`app.py`) has five sections:

1. **Data Overview:** total transactions, fraud count, fraud percentage
2. **Class Distribution:** bar chart of legitimate vs fraud
3. **Train & Evaluate Model:** accuracy, ROC-AUC, confusion matrix, classification report, ROC curve
4. **Feature Importance:** top 15 features
5. **Predict a Transaction:** choose a test transaction and see Fraud / Legitimate with the fraud probability

> **Note:** The deployed app uses the 50,000-row sample (`creditcard_sample.csv`) with a smaller model to fit free-hosting memory limits, so its numbers are lower than the full-dataset results above. The full-dataset results are in the notebook.

---

## 📁 Project Structure

```
creditcard_Fraud-detection/
│
├── credit_card_fraud_detection.ipynb   # Full notebook (EDA, SMOTE, model, evaluation)
├── app.py                              # Streamlit web application
├── creditcard_sample.csv               # 50,000-row sample used by the app
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Python version for deployment
└── README.md
```

---

## ▶️ How to Run

### Option 1: Google Colab (full dataset)
1. Click the **Open in Colab** badge at the top.
2. Run the cells in order. The notebook downloads the full dataset from Kaggle (or upload `creditcard.csv` manually).
3. Use **Runtime → Run all**.

### Option 2: Run the Streamlit app locally
```bash
git clone https://github.com/srijakothakonda/creditcard_Fraud-detection.git
cd creditcard_Fraud-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Limitations

- `V1–V28` are anonymized PCA components, so features cannot be interpreted in business terms.
- No hyperparameter tuning or cross-validation yet.
- The default threshold of 0.5 is used for the reported metrics.
- The model is trained on a static historical dataset; real fraud patterns change over time.

---

## 🚀 Future Improvements

- Compare with Logistic Regression, Random Forest, XGBoost, and LightGBM
- Hyperparameter tuning with Stratified K-Fold cross-validation
- Threshold tuning based on business cost
- Model explainability with SHAP
- Time-based train/test split
- Real-time scoring through an API and cloud deployment
- Drift monitoring and scheduled retraining

---

## 👩‍💻 Author

**Srija Kothakonda**
B.Tech (CS & AI), SR University

**GitHub:** [srijakothakonda](https://github.com/srijakothakonda)
