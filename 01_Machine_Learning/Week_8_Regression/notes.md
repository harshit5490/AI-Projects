# Week 9 – Regression

---

# Learning Objectives

By the end of this week, you should be able to:

- Understand Regression problems
- Build Linear Regression models
- Understand Gradient Descent
- Train and evaluate regression models
- Build Multiple Linear Regression models
- Create Polynomial Regression models
- Prevent Overfitting using Regularization
- Use Ridge, Lasso, and ElasticNet Regression

---

# Folder Structure

```
Week_9_Regression/
│
├── theory.md
├── notes.md
│
├── 01_simple_linear_regression.py
├── 02_gradient_descent_demo.py
├── 03_train_test_split.py
├── 04_linear_regression_model.py
├── 05_model_evaluation.py
├── 06_multiple_linear_regression.py
├── 07_polynomial_regression.py
├── 08_regularization.py
│
├── datasets/
│   ├── Salary_Data.csv
│   └── House_Price.csv
│
├── outputs/
│   ├── regression_line.png
│   ├── polynomial_curve.png
│   └── evaluation_results.txt
│
└── mini_project/
    ├── employee_salary_prediction.py
    └── README.md
```

---

# 1. Regression

Regression predicts **continuous numerical values**.

Examples:

- House Price Prediction
- Salary Prediction
- Sales Forecasting
- Temperature Prediction

Regression ≠ Classification

Regression predicts numbers.

Classification predicts categories.

---

# 2. Simple Linear Regression

Uses **one independent variable**.

Equation

```
y = mx + b
```

Where

- x → Independent Variable
- y → Dependent Variable
- m → Slope
- b → Intercept

Example

```
Hours Studied
        │
        ▼
     Marks
```

---

# 3. Model Training Workflow

```
Dataset
   │
   ▼
Train-Test Split
   │
   ▼
Create Model
   │
   ▼
Train Model
   │
   ▼
Prediction
   │
   ▼
Evaluation
```

---

# 4. Gradient Descent

Gradient Descent finds the **best values of slope and intercept**.

Goal

```
Minimize Error
```

Hyperparameters

- Learning Rate
- Number of Iterations

Too small learning rate

- Slow learning

Too large learning rate

- Overshoots the minimum

---

# 5. Train-Test Split

Purpose

Evaluate model performance on unseen data.

Typical split

```
80% → Training

20% → Testing
```

Python

```python
from sklearn.model_selection import train_test_split
```

---

# 6. Linear Regression in Scikit-Learn

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 7. Model Evaluation Metrics

### MAE

Mean Absolute Error

```
Lower is Better
```

Easy to interpret.

---

### MSE

Mean Squared Error

Squares the errors.

Penalizes large mistakes.

```
Lower is Better
```

---

### RMSE

Root Mean Squared Error

Square root of MSE.

Same unit as target variable.

```
Lower is Better
```

---

### R² Score

Measures how well the model explains the variance.

Range

```
0 → Poor

1 → Perfect
```

Higher is better.

---

# 8. Multiple Linear Regression

Uses **multiple independent variables**.

Equation

```
y = b + m₁x₁ + m₂x₂ + m₃x₃ + ...
```

Example

House Price

↓

- Area
- Bedrooms
- Bathrooms
- Age

---

# 9. Model Coefficients

Each feature has its own coefficient.

Positive coefficient

Feature increases prediction.

Negative coefficient

Feature decreases prediction.

---

# 10. Polynomial Regression

Used when the relationship is **non-linear**.

Instead of

```
y = mx + b
```

Model learns

```
y = b + m₁x + m₂x² + m₃x³
```

Created using

```python
PolynomialFeatures
```

Polynomial Regression is still **Linear Regression** trained on transformed features.

---

# 11. Polynomial Features

Degree 2

```
x

↓

x

x²
```

Degree 3

```
x

↓

x

x²

x³
```

---

# 12. Pipeline Order

Correct preprocessing order

```
Raw Data
    │
    ▼
Missing Values
    │
    ▼
Encoding
    │
    ▼
Polynomial Features
    │
    ▼
Scaling
    │
    ▼
Train Model
```

In production, fit preprocessing only on the training data using an `sklearn.pipeline.Pipeline` to avoid data leakage.

---

# 13. Overfitting

Training Accuracy

```
99%
```

Testing Accuracy

```
65%
```

Model memorizes the training data.

Poor generalization.

---

# 14. Regularization

Purpose

Prevent overfitting.

Three types

```
Ridge

Lasso

ElasticNet
```

---

# 15. Ridge Regression (L2)

Characteristics

- Shrinks coefficients
- Keeps all features
- Useful when features are highly correlated

Python

```python
from sklearn.linear_model import Ridge
```

---

# 16. Lasso Regression (L1)

Characteristics

- Shrinks coefficients
- Can make coefficients exactly zero
- Performs automatic feature selection

Python

```python
from sklearn.linear_model import Lasso
```

---

# 17. ElasticNet

Combination of

```
L1 + L2
```

Advantages

- Feature selection
- Handles correlated features

Python

```python
from sklearn.linear_model import ElasticNet
```

---

# 18. Alpha (Regularization Strength)

Small Alpha

```
Almost Linear Regression
```

Large Alpha

```
Strong Regularization

Possible Underfitting
```

---

# 19. Comparison

| Model | Removes Features | Reduces Coefficients |
|--------|------------------|----------------------|
| Linear Regression | ❌ | ❌ |
| Ridge | ❌ | ✅ |
| Lasso | ✅ | ✅ |
| ElasticNet | Partial | ✅ |

---

# Important Interview Questions

### Q1 Why do we use Train-Test Split?

To evaluate performance on unseen data.

---

### Q2 Which metric is easiest to understand?

MAE

---

### Q3 Why is RMSE preferred over MSE?

RMSE is in the original unit of the target variable.

---

### Q4 Difference between Simple and Multiple Linear Regression?

Simple uses one feature.

Multiple uses two or more features.

---

### Q5 Is Polynomial Regression a separate algorithm?

No.

It is Linear Regression trained on Polynomial Features.

---

### Q6 What causes Overfitting?

- Complex models
- Too many features
- Noise
- High polynomial degree
- Small datasets

---

### Q7 Difference between Ridge and Lasso?

Ridge shrinks coefficients.

Lasso shrinks and removes unnecessary features.

---

### Q8 Which regression performs feature selection?

Lasso Regression.

---

### Q9 What is Alpha?

Regularization strength.

---

### Q10 Which regression is best for highly correlated features?

Ridge Regression.

---

# Python Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

# Mini Project

## Employee Salary Prediction

Features

- Experience
- Education
- Skill Score

Target

- Salary

Tasks

- Build Simple Linear Regression
- Build Multiple Linear Regression
- Try Polynomial Regression
- Compare Ridge, Lasso, and ElasticNet
- Evaluate using MAE, MSE, RMSE, and R² Score

---

# Week 9 Summary

✅ Regression Fundamentals

✅ Simple Linear Regression

✅ Gradient Descent

✅ Train-Test Split

✅ Model Evaluation

✅ Multiple Linear Regression

✅ Polynomial Regression

✅ Polynomial Features

✅ Overfitting

✅ Ridge Regression

✅ Lasso Regression

✅ ElasticNet

✅ Regularization

---

# Outcome

After completing Week 9, you can:

- Build and evaluate regression models
- Choose the correct regression algorithm
- Handle linear and non-linear relationships
- Prevent overfitting using regularization
- Explain regression concepts confidently in interviews
- Apply regression techniques to real-world prediction problems