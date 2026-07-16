# Week 10 – Classification

---

# Learning Objectives

By the end of this week, you will understand:

- What Classification is
- Difference between Regression and Classification
- Types of Classification Problems
- Logistic Regression
- Sigmoid Function
- Decision Boundary
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score
- Precision-Recall Curve
- Average Precision (AP)
- Choosing the right evaluation metric

---

# 1. Classification

## Definition

Classification is a supervised machine learning algorithm used to predict **categorical values**.

Unlike regression, which predicts continuous numbers, classification predicts categories.

Examples:

- Spam / Not Spam
- Fraud / Not Fraud
- Employee Leave / Stay
- Disease / Healthy

---

## Types of Classification

### Binary Classification

Only two classes.

Examples

- Yes / No
- 0 / 1
- Leave / Stay

Algorithms

- Logistic Regression
- SVM
- Decision Tree
- Random Forest

---

### Multi-Class Classification

More than two classes.

Examples

- Cat
- Dog
- Horse

---

### Multi-Label Classification

One sample can belong to multiple classes.

Example

Movie Genres

- Action
- Comedy
- Drama

---

# Regression vs Classification

| Regression | Classification |
|------------|---------------|
| Predicts continuous values | Predicts categories |
| Output = Number | Output = Class |
| Example: House Price | Example: Spam Detection |

---

# Logistic Regression

Although the name contains "Regression", it is a **classification algorithm**.

It predicts the **probability** of belonging to the positive class.

---

## Logistic Regression Equation

Linear Equation

\[
z = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n
\]

The output is then passed through the Sigmoid Function.

---

# Sigmoid Function

Formula

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

Output Range

```
0 → 1
```

Properties

- Converts any real number into probability
- Smooth S-shaped curve
- Used in Binary Classification

---

# Decision Boundary

Converts probability into class.

Default Threshold

```
Probability >= 0.5

↓

Class = 1

Probability < 0.5

↓

Class = 0
```

---

## Can Threshold be Changed?

Yes.

Example

Threshold = 0.7

Model becomes more strict.

Threshold = 0.3

Model predicts more positives.

---

## Real-world Examples

Cancer Detection

Lower threshold

Higher Recall

Loan Approval

Higher threshold

Higher Precision

---

# Confusion Matrix

Compares

Actual Values

vs

Predicted Values

| | Predicted Positive | Predicted Negative |
|---|---:|---:|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

---

## True Positive (TP)

Actual Positive

Predicted Positive

Correct prediction.

---

## True Negative (TN)

Actual Negative

Predicted Negative

Correct prediction.

---

## False Positive (FP)

Actual Negative

Predicted Positive

Type-I Error

---

## False Negative (FN)

Actual Positive

Predicted Negative

Type-II Error

---

# Accuracy

Definition

Overall correctness of the model.

Formula

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

Use When

- Dataset is balanced

Limitation

Can be misleading for imbalanced datasets.

---

# Precision

Definition

Out of all predicted positives,

how many are actually positive?

Formula

\[
Precision=\frac{TP}{TP+FP}
\]

Use When

False Positives are costly.

Examples

- Spam Detection
- Loan Approval

---

# Recall

Definition

Out of all actual positives,

how many were correctly identified?

Formula

\[
Recall=\frac{TP}{TP+FN}
\]

Use When

False Negatives are costly.

Examples

- Cancer Detection
- Fraud Detection
- Employee Attrition

---

# F1 Score

Definition

Balances Precision and Recall.

Formula

\[
F1=\frac{2\times Precision\times Recall}{Precision+Recall}
\]

Uses Harmonic Mean.

Range

```
0 → 1
```

Use When

- Dataset is imbalanced
- Need balance between Precision and Recall

---

# ROC Curve

Full Form

Receiver Operating Characteristic

Plots

```
True Positive Rate

vs

False Positive Rate
```

---

## True Positive Rate

Same as Recall

\[
TPR=\frac{TP}{TP+FN}
\]

---

## False Positive Rate

\[
FPR=\frac{FP}{FP+TN}
\]

---

## ROC Interpretation

Closer to

Top-left Corner

↓

Better Model

Diagonal Line

↓

Random Guess

---

# AUC Score

Area Under ROC Curve

Range

```
0 → 1
```

Interpretation

| AUC | Meaning |
|-----|---------|
|1.00|Perfect|
|0.90+|Excellent|
|0.80+|Good|
|0.70+|Fair|
|0.50|Random Guess|

Higher is Better.

---

# Precision-Recall Curve

Plots

```
Precision

vs

Recall
```

instead of

```
TPR

vs

FPR
```

Best for

Imbalanced Datasets

Examples

- Fraud Detection
- Employee Attrition
- Disease Prediction

---

# Average Precision (AP)

Area Under Precision-Recall Curve.

Range

```
0 → 1
```

Higher is Better.

---

# Choosing the Right Metric

| Situation | Best Metric |
|-----------|------------|
| Balanced Dataset | Accuracy |
| False Positives are costly | Precision |
| False Negatives are costly | Recall |
| Need balance between Precision & Recall | F1 Score |
| Balanced Dataset Comparison | ROC-AUC |
| Imbalanced Dataset | PR Curve / Average Precision |

---

# Python Functions

## Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## Predict Class

```python
y_pred = model.predict(X_test)
```

---

## Predict Probability

```python
y_prob = model.predict_proba(X_test)
```

---

## Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
```

---

## Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

---

## Precision

```python
from sklearn.metrics import precision_score

precision_score(y_test, y_pred)
```

---

## Recall

```python
from sklearn.metrics import recall_score

recall_score(y_test, y_pred)
```

---

## F1 Score

```python
from sklearn.metrics import f1_score

f1_score(y_test, y_pred)
```

---

## ROC-AUC

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_prob)
```

---

## ROC Curve

```python
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y_test, y_prob)
```

---

## Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_test, y_prob)
```

---

## Average Precision

```python
from sklearn.metrics import average_precision_score

average_precision_score(y_test, y_prob)
```

---

# Interview Questions

### Q1. Difference between Regression and Classification?

Regression predicts continuous values.

Classification predicts categories.

---

### Q2. Why is Logistic Regression called Regression?

Because it calculates a linear regression equation internally before applying the Sigmoid function to produce probabilities for classification.

---

### Q3. Why do we use Sigmoid Function?

To convert linear outputs into probabilities between 0 and 1.

---

### Q4. What is the default Decision Threshold?

```
0.5
```

---

### Q5. Can we change the threshold?

Yes.

Using `predict_proba()` we can define a custom threshold based on business requirements.

---

### Q6. What is the difference between Precision and Recall?

Precision measures prediction quality.

Recall measures how many actual positives were found.

---

### Q7. Why do we need F1 Score?

Because Accuracy may be misleading for imbalanced datasets, and F1 balances Precision and Recall.

---

### Q8. Difference between ROC Curve and PR Curve?

ROC

- Balanced datasets
- TPR vs FPR

PR Curve

- Imbalanced datasets
- Precision vs Recall

---

### Q9. What does AUC represent?

Overall ability of the classifier to distinguish between positive and negative classes.

---

### Q10. Which metric is best for Employee Attrition?

Usually

- Recall (to avoid missing employees likely to leave)
- F1 Score (to balance Recall and Precision)
- PR-AUC (if attrition cases are rare)

---

# Key Takeaways

- Classification predicts categories, not numbers.
- Logistic Regression outputs probabilities using the Sigmoid function.
- Decision Boundary converts probabilities into classes.
- Confusion Matrix is the foundation of all classification metrics.
- Accuracy is useful only for balanced datasets.
- Precision focuses on False Positives.
- Recall focuses on False Negatives.
- F1 Score balances Precision and Recall.
- ROC-AUC evaluates class separation across all thresholds.
- PR Curve and Average Precision are preferred for imbalanced datasets.

---

# Week 10 Summary

✅ Classification Fundamentals

✅ Binary, Multi-class & Multi-label Classification

✅ Logistic Regression

✅ Sigmoid Function

✅ Decision Boundary

✅ Confusion Matrix

✅ Accuracy

✅ Precision

✅ Recall

✅ F1 Score

✅ ROC Curve

✅ AUC Score

✅ Precision-Recall Curve

✅ Average Precision

✅ Choosing the Correct Evaluation Metric

---

# Next Week

**Week 11 – Model Evaluation**

Topics:

- Train-Test Split
- Overfitting vs Underfitting
- Cross Validation
- K-Fold Cross Validation
- Stratified K-Fold
- Bias-Variance Tradeoff
- Learning Curves
- Validation Curves
- Hyperparameter Tuning
- Grid Search
- Random Search
- Model Selection