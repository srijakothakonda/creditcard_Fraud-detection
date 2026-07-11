# Credit Card Fraud Detection using SMOTE and Random Forest

## Project Overview

This project aims to detect fraudulent credit card transactions using Machine Learning. Since fraud transactions are extremely rare compared to legitimate transactions, the dataset is highly imbalanced.

To solve this problem, SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the training data, and a Random Forest Classifier is trained to identify fraudulent transactions.

## Dataset

Dataset: Credit Card Fraud Detection Dataset

Features:

* Time
* V1 to V28 (PCA-transformed features)
* Amount
* Class

Target Variable:

* 0 = Legitimate Transaction
* 1 = Fraudulent Transaction

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib
* Seaborn
* Streamlit

## Project Workflow

1. Load Dataset
2. Data Preprocessing
3. Feature Scaling
4. Train-Test Split
5. Apply SMOTE
6. Train Random Forest Model
7. Evaluate Performance
8. Deploy Using Streamlit

## Machine Learning Model

Algorithm:

* Random Forest Classifier

Reason:

* Handles large datasets effectively
* Reduces overfitting
* Provides strong classification performance

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

## Results

The model successfully identifies fraudulent transactions while maintaining strong performance on legitimate transactions.

## Future Improvements

* XGBoost Implementation
* LightGBM Implementation
* Real-Time Fraud Detection
* Cloud Deployment

## Author
DEMO LINK:https://creditcardfraud-detection-5wkfuzithcuygwydnkjvin.streamlit.app/

Srija Kothakonda
B.Tech (CS & AI)
SR University
