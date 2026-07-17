# Week 11 - Model Evaluation (Advanced)

---

# Overview

This week focused on understanding how to evaluate machine learning models, detect underfitting and overfitting, and automatically find the best hyperparameters.

Unlike previous weeks where the goal was to build models, this week focused on improving model performance and selecting the best model.

---

# Module 1 : Bias and Variance

## Bias

Bias is the error caused by making overly simple assumptions.

Characteristics

- Model is too simple
- High training error
- High testing error
- Underfitting

Example

- Linear Regression on highly non-linear data

---

## Variance

Variance is the error caused by the model learning too many details from training data.

Characteristics

- Very low training error
- High testing error
- Overfitting

Example

- Decision Tree with very large depth

---

## Bias-Variance Tradeoff

Goal

Find the balance between

- Low Bias
- Low Variance

A good model should generalize well to unseen data.

---

# Module 2 : Underfitting vs Overfitting

## Underfitting

Definition

Model is too simple to capture patterns.

Symptoms

- Low Training Accuracy
- Low Testing Accuracy

Solution

- Increase model complexity
- Add features
- Reduce regularization

---

## Good Fit

Characteristics

- High Training Accuracy
- High Testing Accuracy
- Small difference between them

---

## Overfitting

Definition

Model memorizes training data.

Symptoms

- Very High Training Accuracy
- Low Testing Accuracy

Solution

- Cross Validation
- Regularization
- Reduce model complexity
- More training data
- Early Stopping

---

# Module 3 : Learning Curve

Definition

A Learning Curve shows model performance as the amount of training data increases.

X-axis

Training Samples

Y-axis

Accuracy

Two Curves

- Training Score
- Validation Score

Uses

- Detect Underfitting
- Detect Overfitting
- Determine if more data will help

Interpretation

Underfitting

Both curves are low.

Good Fit

Both curves converge at a high accuracy.

Overfitting

Training score is very high but validation score is much lower.

---

# Module 4 : Cross Validation

Purpose

Provides a more reliable estimate of model performance.

Instead of using one train-test split,

the dataset is divided into K folds.

Example

5-Fold Cross Validation

Step 1

Split dataset into 5 equal parts.

Step 2

Train on 4 folds.

Step 3

Test on remaining fold.

Repeat until every fold has been used as validation.

Final Score

Average of all validation scores.

Advantages

- Better model evaluation
- Uses entire dataset
- Reduces overfitting caused by one train-test split

Common Values

- cv = 5
- cv = 10

---

# Module 5 : Stratified Cross Validation

Problem

Normal K-Fold may produce imbalanced class distributions.

Solution

Stratified K-Fold

It keeps the class proportion almost identical in every fold.

Example

Dataset

90% No

10% Yes

Each fold

90% No

10% Yes

Used mostly for

Classification problems.

---

# Module 6 : Hyperparameter Tuning

Hyperparameters

Values chosen before training.

Examples

Decision Tree

- max_depth
- min_samples_split

Random Forest

- n_estimators
- max_depth

Logistic Regression

- C

SVM

- C
- gamma

Purpose

Find the combination that gives the highest validation performance.

---

# Module 7 : Validation Curve

Definition

A Validation Curve shows model performance as a hyperparameter changes.

Learning Curve

Changes training size.

Validation Curve

Changes hyperparameter value.

Example

Decision Tree

max_depth

1

↓

10

Graph

- Blue → Training Accuracy
- Orange → Validation Accuracy

Interpretation

Small max_depth

Underfitting

Best validation accuracy

Optimal model

Very large max_depth

Overfitting

Uses

- Choose best hyperparameter
- Detect overfitting
- Detect underfitting

---

# Module 8 : GridSearchCV

Definition

GridSearchCV automatically tests every combination of hyperparameters using Cross Validation.

Workflow

Choose Hyperparameters

↓

Create Parameter Grid

↓

GridSearchCV

↓

Cross Validation

↓

Average Score

↓

Best Hyperparameters

Important Parameters

estimator

Machine Learning model

param_grid

Dictionary of hyperparameters

cv

Number of folds

scoring

Evaluation metric

Useful Attributes

best_params_

Returns best hyperparameters.

best_score_

Returns highest cross-validation score.

best_estimator_

Returns trained model with best parameters.

Advantages

- Automatic tuning
- Reliable
- Finds best combination

Disadvantages

- Computationally expensive
- Slow for large search spaces

---

# Module 9 : RandomizedSearchCV

Definition

RandomizedSearchCV randomly selects hyperparameter combinations instead of testing every combination.

Difference from Grid Search

Grid Search

Tests all combinations.

Random Search

Tests only random combinations.

Important Parameter

n_iter

Number of random combinations to test.

Advantages

- Much faster
- Works well for large search spaces
- Nearly same performance as Grid Search

Disadvantages

- May miss the absolute best combination
- Randomness affects results

---

# GridSearchCV vs RandomizedSearchCV

| GridSearchCV | RandomizedSearchCV |
|---------------|-------------------|
| Tests every combination | Tests random combinations |
| Slower | Faster |
| Higher computational cost | Lower computational cost |
| Best for small search space | Best for large search space |

---

# Important sklearn Functions

```python
from sklearn.model_selection import

train_test_split
cross_val_score
validation_curve
GridSearchCV
RandomizedSearchCV
```

---

# Important Attributes

```python
grid.best_params_

grid.best_score_

grid.best_estimator_
```

---

# Common Evaluation Metrics Used

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Regression

- MAE
- MSE
- RMSE
- R² Score

---

# Interview Questions

## Q1. What is Cross Validation?

A technique that evaluates a model by training and testing on multiple folds of the dataset.

---

## Q2. Why is Cross Validation better than Train-Test Split?

Because every sample is used for both training and validation, producing a more reliable performance estimate.

---

## Q3. Difference between Learning Curve and Validation Curve?

Learning Curve

Changes training data size.

Validation Curve

Changes hyperparameter value.

---

## Q4. What is Hyperparameter Tuning?

The process of finding the best values for model hyperparameters before final training.

---

## Q5. What is GridSearchCV?

A method that tests every hyperparameter combination using Cross Validation.

---

## Q6. What is RandomizedSearchCV?

A method that tests only a random subset of hyperparameter combinations.

---

## Q7. When should you use GridSearchCV?

When the search space is small.

---

## Q8. When should you use RandomizedSearchCV?

When the search space is large.

---

## Q9. What does `best_estimator_` return?

The trained model with the best hyperparameters.

---

## Q10. What does `n_iter` mean in RandomizedSearchCV?

The number of random hyperparameter combinations that will be evaluated.

---

# Key Takeaways

- Understand Bias and Variance.
- Detect Underfitting and Overfitting.
- Use Learning Curves to analyze training size effects.
- Use Validation Curves to tune a single hyperparameter.
- Use Cross Validation for reliable model evaluation.
- Use Stratified K-Fold for imbalanced classification datasets.
- Use GridSearchCV for exhaustive hyperparameter tuning.
- Use RandomizedSearchCV for faster tuning on large search spaces.
- Prefer the model with the highest validation performance rather than the highest training accuracy.

---

# Week 11 Summary

By the end of Week 11, you can:

✅ Explain Bias-Variance Tradeoff

✅ Identify Underfitting and Overfitting

✅ Interpret Learning Curves

✅ Interpret Validation Curves

✅ Apply K-Fold and Stratified K-Fold Cross Validation

✅ Tune Machine Learning models using GridSearchCV

✅ Tune Machine Learning models using RandomizedSearchCV

✅ Select the best model using Cross Validation instead of relying only on training accuracy