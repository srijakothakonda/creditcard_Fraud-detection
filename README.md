
# Credit Card Fraud Detection 🚀

## 📌 Overview

This project builds a machine learning model to detect fraudulent credit card transactions using **imbalanced dataset handling techniques**. Fraud detection is crucial because fraud transactions are rare but costly. The dataset is heavily skewed toward non-fraudulent transactions, which requires special techniques like **SMOTE** to balance the data before training.


## 🛠️ Technologies Used

Python 3.x**
Scikit-learn** (Random Forest, preprocessing, metrics)
Pandas & NumPy** (data manipulation)
Imbalanced-learn (SMOTE)** for handling class imbalance
Matplotlib & Seaborn** for visualization

---

## ⚙️ Features

* Data preprocessing and identification of categorical vs numeric features
* Handling class imbalance using **SMOTE** to oversample minority (fraud) class
* Model training using **Random Forest Classifier**
* Evaluation using **Accuracy, Precision, Recall, F1-score**
* Visualization of **confusion matrix** for performance analysis

---

## 📊 Results

| Metric    | Value    |
| --------- | -------- |
| Accuracy  | 99.35%   |
| Precision | 94.74%   |
| Recall    | 60.00% ⭐ |
| F1 Score  | 73.47%   |

> **Note:** Recall is critical in fraud detection because missing fraudulent transactions can be very costly, even if the overall accuracy is high.

**Confusion Matrix:**

```
[[1969    1]
 [  12   18]]
```

  True Negatives (1969):** Correctly predicted non-fraud
  False Positives (1):** Non-fraud predicted as fraud
  False Negatives (12):** Fraud missed by the model
  True Positives (18):** Fraud correctly detected

---

## ▶️ How to Run

1. Open this notebook in **Google Colab**
2. **Upload the CSV dataset** when prompted
3. Run all cells sequentially
4. View **model performance metrics and confusion matrix**

