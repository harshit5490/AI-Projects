# Customer Churn Prediction — End-to-End Machine Learning Project

## 1. Project Overview

This project builds an end-to-end machine learning pipeline for predicting
customer churn using the Telco Customer Churn dataset.

The project covers the complete ML lifecycle including:

- Data exploration
- Data preprocessing
- Feature engineering
- Model training
- Ensemble learning
- Model evaluation
- Hyperparameter tuning
- Model persistence
- Prediction on new customer data

---

## 2. Business Problem

Customer churn occurs when customers stop using a company's services.

The objective of this project is to build a machine learning model capable
of identifying customers who are likely to churn so that appropriate
retention strategies can be applied.

---

## 3. Dataset

Dataset: Telco Customer Churn

The dataset contains customer information including:

- Demographics
- Account information
- Internet services
- Contract information
- Payment methods
- Monthly charges
- Total charges
- Customer tenure
- Churn status

Target variable:

Churn

0 = No Churn  
1 = Churn

---

## 4. Project Workflow

Raw Dataset
    ↓
Exploratory Data Analysis
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Train-Test Split
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Model Comparison
    ↓
Model Evaluation
    ↓
Hyperparameter Tuning
    ↓
Final Model Selection
    ↓
Prediction Pipeline

---

## 5. Machine Learning Models

The following models were trained and compared:

- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Stacking Classifier

Model selection was primarily based on F1 Score.

---

## 6. Best Model

The best-performing baseline model was:

Gradient Boosting Classifier

Test Performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.7953 |
| Precision | 0.6378 |
| Recall | 0.5321 |
| F1 Score | 0.5802 |

Hyperparameter tuning was also performed using GridSearchCV.

The tuned model achieved an F1 Score of 0.5793, which did not outperform
the baseline model. Therefore, the baseline Gradient Boosting model was
retained as the final model.

---

## 7. Model Evaluation

The final model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

---

## 8. Hyperparameter Tuning

GridSearchCV with Stratified K-Fold Cross Validation was used for
hyperparameter optimization.

Parameters explored included:

- n_estimators
- learning_rate
- max_depth
- min_samples_split
- min_samples_leaf

The tuned model was compared against the baseline model before final
model selection.

---

## 9. Project Structure

Week_15_End_to_End_ML_Project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── images/
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hyperparameter_tuning.py
│   ├── predict.py
│   └── utils.py
│
├── README.md
├── notes.md
├── requirements.txt
└── .gitignore

---

## 10. Prediction Pipeline

The prediction pipeline loads the final trained model and scaler and
generates predictions for new customer data.

Example output:

Prediction : Churn  
Probability: 71.00%

---

## 11. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- Joblib

---

## 12. How to Run

### Data preprocessing

python src/data_preprocessing.py

### Feature engineering

python src/feature_engineering.py

### Model training

python src/train.py

### Model evaluation

python src/evaluate.py

### Hyperparameter tuning

python src/hyperparameter_tuning.py

### Prediction

python src/predict.py

---

## 13. Key Learnings

This project demonstrates:

- Building modular machine learning pipelines
- Preventing data leakage
- Handling imbalanced classification data
- Comparing multiple machine learning algorithms
- Evaluating models using multiple metrics
- Hyperparameter optimization
- Model persistence
- Reusable prediction pipelines
- Production-style ML project organization

---

## 14. Future Improvements

Possible improvements include:

- Scikit-learn Pipeline / ColumnTransformer
- Automated preprocessing for raw prediction input
- SHAP-based model explainability
- API deployment using FastAPI
- Docker containerization
- Experiment tracking
- Model monitoring
- CI/CD