# Week 8 – Supervised Learning

## Objective

This week introduces the fundamentals of Supervised Learning.

By the end of this week, you will be able to:

- Understand Supervised Learning
- Differentiate Classification and Regression
- Understand Evaluation Metrics
- Split data correctly
- Use the Scikit-Learn workflow
- Build your first Machine Learning model

---

# What is Supervised Learning?

Supervised Learning is a type of Machine Learning where the model learns from labeled data.

Input (X)
↓

Model Learns

↓

Output (y)

Example

Age, Salary, Experience

↓

Predict Employee Attrition

---

# Types of Supervised Learning

There are two main types.

## 1. Classification

Predicts categories.

Examples

- Employee Attrition
- Spam Detection
- Disease Prediction
- Loan Approval
- Customer Churn

Output

Yes / No

Spam / Not Spam

Approved / Rejected

Cat / Dog

---

### Binary Classification

Only two classes.

Examples

- Yes / No
- True / False
- 0 / 1

---

### Multi-Class Classification

More than two classes.

Example

Cat

Dog

Horse

Cow

---

### Multi-Label Classification

One record belongs to multiple classes.

Example

Movie

Action

Comedy

Adventure

---

## 2. Regression

Regression predicts continuous numerical values.

Examples

- House Price
- Salary
- Temperature
- Sales
- Stock Price

Output

250000

32.5

45000

18.4

---

# Classification vs Regression

| Classification | Regression |
|----------------|------------|
| Predict Category | Predict Number |
| Discrete Output | Continuous Output |
| Accuracy | MAE |
| Precision | MSE |
| Recall | RMSE |
| F1 Score | R² Score |

---

# Evaluation Metrics

Evaluation metrics measure model performance.

Without metrics, we cannot determine whether a model performs well.

---

## Regression Metrics

Used for numerical prediction.

- MAE
- MSE
- RMSE
- R² Score

---

## Classification Metrics

Used for categorical prediction.

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Important Note

Accuracy is not always the best metric.

Example

Employee Attrition

Stay = 84%

Leave = 16%

If a model predicts every employee stays,

Accuracy = 84%

But the model completely fails to identify employees who leave.

Therefore,

Precision

Recall

F1 Score

become more important.

---

# Train-Test Split

Purpose

Evaluate the model on unseen data.

Dataset

↓

Training Data

↓

Testing Data

---

## Common Split Ratios

80 : 20

70 : 30

90 : 10

Most commonly used

80 : 20

---

# Training Set

Used to train the model.

The model learns from this data.

---

# Testing Set

Used only for evaluation.

The model never sees this data during training.

---

# Why Split the Data?

If we train and test on the same data,

the model may memorize instead of learning.

This leads to Overfitting.

---

# Random State

Example

random_state = 42

Purpose

Produces the same train-test split every time.

Useful for reproducibility.

---

# Stratification

Used mainly for classification.

Purpose

Maintain the same class distribution in both train and test datasets.

Example

Original Dataset

84% Stay

16% Leave

Train Dataset

84% Stay

16% Leave

Test Dataset

84% Stay

16% Leave

Implementation

stratify = y

---

# Data Leakage

One of the biggest mistakes in Machine Learning.

Wrong

Scale Entire Dataset

↓

Train-Test Split

Correct

Train-Test Split

↓

Fit Scaler on Training Data

↓

Transform Training Data

↓

Transform Test Data

---

# Scikit-Learn

Scikit-Learn is the most popular Machine Learning library in Python.

Provides

- Algorithms
- Preprocessing
- Evaluation
- Model Selection
- Feature Engineering
- Pipelines

---

# Standard Machine Learning Workflow

1. Import Libraries

↓

2. Load Dataset

↓

3. Separate Features and Target

↓

4. Train-Test Split

↓

5. Create Model

↓

6. Train Model

↓

7. Predict

↓

8. Evaluate

---

# Important Methods

## fit()

Used to train the model.

Example

model.fit(X_train, y_train)

---

## predict()

Predict unseen data.

Example

model.predict(X_test)

---

## predict_proba()

Returns probability for each class.

Example

[[0.95 0.05]]

Meaning

95% Class 0

5% Class 1

---

## score()

Quick evaluation.

Classification

Returns Accuracy

Regression

Returns R² Score

---

# First Machine Learning Model

Steps

Load Dataset

↓

Split Dataset

↓

Create Model

↓

Train Model

↓

Predict

↓

Evaluate

---

# K-Nearest Neighbors (KNN)

Used as the first algorithm because it is simple.

Workflow

Create Model

↓

model.fit()

↓

model.predict()

↓

accuracy_score()

---

# Best Practices

✓ Always split data before training.

✓ Use random_state for reproducibility.

✓ Use stratify for classification.

✓ Never train on testing data.

✓ Never scale before train-test split.

✓ Keep testing data completely unseen.

---

# Real-World Examples

| Problem | Type |
|----------|------|
| Employee Attrition | Classification |
| House Price Prediction | Regression |
| Customer Churn | Classification |
| Salary Prediction | Regression |
| Loan Approval | Classification |
| Sales Forecasting | Regression |
| Disease Prediction | Classification |
| Temperature Prediction | Regression |

---

# Interview Questions

1. What is Supervised Learning?

2. Difference between Classification and Regression?

3. What is Train-Test Split?

4. Why do we split data?

5. What is random_state?

6. What is stratification?

7. What is data leakage?

8. What does fit() do?

9. What does predict() do?

10. What does predict_proba() return?

11. What does score() return?

12. Why is Accuracy not always the best metric?

13. What are the steps of the Machine Learning workflow?

14. Why should scaling happen after Train-Test Split?

15. What is KNN?

---

# Key Takeaways

- Supervised Learning uses labeled data.
- Classification predicts categories.
- Regression predicts continuous values.
- Different metrics are used for different problem types.
- Always split data before training.
- Avoid data leakage.
- Scikit-Learn follows a standard workflow.
- fit() trains the model.
- predict() predicts unseen data.
- score() evaluates the model.
- A complete ML pipeline consists of loading data, preprocessing, training, prediction, and evaluation.

---

# Week 8 Summary

Lessons Completed

✅ Lesson 1 – Classification vs Regression

✅ Lesson 2 – Evaluation Metrics Overview

✅ Lesson 3 – Train-Test Split

✅ Lesson 4 – Scikit-Learn Workflow

✅ Lesson 5 – First Machine Learning Model

Mini Project

✅ End-to-End Supervised Learning Workflow

Status

✅ Week 8 Completed Successfully