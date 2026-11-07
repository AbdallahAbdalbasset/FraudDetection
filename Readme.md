# Credit Card Fraud Detection 🕵️‍♂️💳

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

## 📌 Project Overview
This repository contains a robust Machine Learning pipeline designed to detect fraudulent credit card transactions. The project tackles a classic binary classification problem characterized by a **highly imbalanced dataset** (where fraudulent transactions represent $< 1\%$ of the total data). 

Following the guidelines:
the primary objective of this project is to optimize the **F1-Score**, ensuring a strong balance between Precision and Recall when identifying fraud.

## 📊 The Data
* **Features:** The dataset consists primarily of anonymous numerical features (resulting from a PCA transformation to protect user privacy), with two explicit features: `Time` and `Amount`.
* **Challenge:** Extreme class imbalance. Traditional accuracy is not a reliable metric here, requiring specialized techniques and threshold optimization.
* **Splits:** Data is divided into `train.csv`, `val.csv`, and `test.csv` for rigorous training, validation, and unbiased testing.

## 🛠️ Methodology & Practices
To combat the imbalance and extract the best possible predictive power, the following practices were implemented:
1. **Data Scaling:** Features were standardized using `StandardScaler`, fit strictly on the training set to prevent data leakage.
2. **Handling Imbalance:** Explored various advanced concepts including **SMOTE** (Synthetic Minority Over-sampling Technique) and **Focal Loss** to penalize misclassifications of the minority class.
3. **Threshold Tuning:** Instead of relying on the default `0.5` classification threshold, the optimal threshold was dynamically calculated for each model by maximizing the F1-Score along the **Precision-Recall Curve** using the validation set.
4. **Ensemble Learning:** Utilized **Soft Voting Classifiers** to combine the probabilistic predictions of multiple diverse models, reducing variance and boosting overall performance.

## 🤖 Models Evaluated
Extensive hyperparameter tuning was conducted across a variety of models:
* Logistic Regression
* Random Forest Classifier
* Multi-Layer Perceptron (MLP)
* XGBoost & Gradient Boosting Classifiers
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)

### The Final Ensemble (Best Performer)
The highest-performing architecture was a **Soft Voting Classifier** combining four models with equal weighting:
1. Random Forest (`n_estimators=200`)
2. Gradient Boosting (`n_estimators=100`)
3. SVC (`probability=True`)
4. KNN (`n_neighbors=5`)

## 📈 Results & Metrics

All models were evaluated on the unseen test set using the optimal threshold discovered during validation. 

| Model | Optimal Threshold | Precision | Recall | F1-Score | ROC-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.247 | 0.765 | 0.773 | 0.769 | 0.980 | 0.9992 |
| **Random Forest** | 0.340 | 0.838 | 0.856 | 0.847 | 0.952 | 0.9995 |
| **Voting (LR + RF)** | 0.446 | 0.879 | 0.825 | 0.851 | 0.978 | 0.9995 |
| **Voting (4 Models)** | **0.245** | **0.848** | **0.866** | **0.857** | **0.971** | **0.9995** |

### Confusion Matrix (Best Model)
```text
[[56848    15]   <-- True Negatives (Legitimate) | False Positives (Flagged as Fraud)
 [   13    84]]  <-- False Negatives (Missed)    | True Positives (Caught Fraud)
```
*The final model successfully caught 84 out of 97 fraudulent transactions, with only 15 false alarms out of nearly 57,000 legitimate transactions.*

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the training script:**
   ```bash
   python src/credit_fraud_train.py --data_path ../data/
   ```
