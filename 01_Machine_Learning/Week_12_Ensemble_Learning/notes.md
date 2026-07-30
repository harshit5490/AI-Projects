# Week 13 -- Ensemble Learning Notes

## Learning Objectives

-   Understand ensemble learning and why combining models improves
    performance.
-   Differentiate **Bagging** and **Boosting**.
-   Learn the working principles, strengths, weaknesses, and use cases
    of:
    -   Decision Tree (baseline)
    -   Random Forest
    -   Extra Trees
    -   AdaBoost
    -   Gradient Boosting
    -   XGBoost
    -   LightGBM
    -   CatBoost

------------------------------------------------------------------------

# 1. What is Ensemble Learning?

Ensemble Learning combines multiple machine learning models to produce a
stronger model.

**Idea:**

Weak Learners + Weak Learners + Weak Learners → Strong Learner

### Benefits

-   Higher accuracy
-   Better generalization
-   Reduced overfitting
-   More stable predictions

------------------------------------------------------------------------

# 2. Types of Ensemble Learning

## A. Bagging (Bootstrap Aggregating)

-   Models are trained independently.
-   Final prediction is based on voting (classification) or averaging
    (regression).
-   Mainly reduces **variance**.

Algorithms: - Random Forest - Extra Trees

## B. Boosting

-   Models are trained sequentially.
-   Each model learns from previous mistakes.
-   Mainly reduces **bias**.

Algorithms: - AdaBoost - Gradient Boosting - XGBoost - LightGBM -
CatBoost

------------------------------------------------------------------------

# 3. Decision Tree (Baseline)

## Advantages

-   Easy to understand
-   Fast to train
-   No feature scaling required

## Disadvantages

-   High variance
-   Prone to overfitting

------------------------------------------------------------------------

# 4. Random Forest

## Working

-   Creates many Decision Trees using bootstrap samples.
-   Trees are trained independently.
-   Uses majority voting.

## Advantages

-   Reduces overfitting
-   Stable performance
-   Good baseline model

## Disadvantages

-   Larger model
-   Does not learn from previous mistakes

------------------------------------------------------------------------

# 5. Extra Trees (Extremely Randomized Trees)

## Difference from Random Forest

-   Uses more randomness.
-   Random split thresholds.
-   Usually faster on large datasets, though performance depends on
    data.

## Advantages

-   Lower variance
-   Strong baseline

## Disadvantages

-   May underperform Random Forest on some datasets.

------------------------------------------------------------------------

# 6. AdaBoost (Adaptive Boosting)

## Core Idea

-   Uses weak learners (Decision Stumps).
-   Increases the weight of misclassified samples.
-   Next model focuses on difficult observations.

## Advantages

-   Learns from mistakes
-   High precision on many datasets

## Disadvantages

-   Sensitive to noisy labels and outliers

------------------------------------------------------------------------

# 7. Gradient Boosting

## Core Idea

Instead of increasing sample weights, each new tree learns the
**residual errors** of previous trees.

Residual = Actual − Prediction

Final Prediction = Tree1 + LearningRate × Tree2 + ...

## Advantages

-   Strong predictive performance
-   Flexible
-   Foundation of modern boosting algorithms

## Disadvantages

-   Sequential training
-   Slower than bagging methods

------------------------------------------------------------------------

# 8. XGBoost (Extreme Gradient Boosting)

## Improvements

-   Regularization (L1/L2)
-   Optimized tree construction
-   Missing value handling
-   Parallelized implementation
-   Tree pruning

## Advantages

-   Excellent performance after tuning
-   Fast implementation
-   Widely used in competitions

## Disadvantages

-   Many hyperparameters
-   Default settings are not always optimal

------------------------------------------------------------------------

# 9. LightGBM

Developed by Microsoft.

## Key Innovations

-   Leaf-wise tree growth
-   Histogram-based learning
-   Gradient-based One-Side Sampling (GOSS)
-   Efficient Feature Bundling (EFB)

## Advantages

-   Fast on large datasets
-   Low memory usage
-   High Recall/F1 on many tabular problems

## Disadvantages

-   Can overfit without proper tuning
-   Speed advantage is most noticeable on large datasets

------------------------------------------------------------------------

# 10. CatBoost

Developed by Yandex.

## Key Innovations

-   Native categorical feature handling
-   Ordered Boosting
-   Symmetric (Oblivious) Trees

## Advantages

-   Minimal preprocessing
-   Excellent for categorical datasets
-   Reduces target leakage

## Disadvantages

-   Native categorical advantage is reduced if data is already one-hot
    encoded.

# 11. Stacking (Stacked Generalization)

## What is Stacking?

Stacking is an ensemble technique where the predictions of multiple
**base models** are used as inputs to another model called the
**meta-model** (or final estimator).

### Architecture

``` text
Training Data
      │
      ├── Random Forest
      ├── XGBoost
      ├── LightGBM
      └── Logistic Regression
            │
Predictions from all base models
            │
        Meta Model
    (e.g., Logistic Regression)
            │
      Final Prediction
```

Unlike Bagging and Boosting, Stacking combines **different types of
models** instead of creating many versions of the same model.

## How Stacking Works

1.  Train multiple base models.
2.  Generate predictions from each base model.
3.  Use those predictions as features.
4.  Train a meta-model on these new features.
5.  The meta-model learns how to combine the strengths of all base
    models.

## Advantages

-   Can achieve higher accuracy than individual models.
-   Combines strengths of different algorithms.
-   Often performs well in machine learning competitions.

## Disadvantages

-   More computationally expensive.
-   More complex to train and interpret.
-   Risk of overfitting if cross-validation is not used correctly.

## Scikit-learn Example

``` python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

estimators = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
```

## Interview Question

**Q:** What is the difference between Stacking and Voting?

**Answer:** - Voting combines predictions directly using majority voting
or probability averaging. - Stacking trains a meta-model to learn the
best way to combine predictions from multiple base models.

------------------------------------------------------------------------

# Bagging vs Boosting

  Feature            Bagging           Boosting
  ------------------ ----------------- ----------------------
  Training           Independent       Sequential
  Main Goal          Reduce Variance   Reduce Bias
  Final Prediction   Voting/Average    Weighted Combination
  Example            Random Forest     Gradient Boosting

------------------------------------------------------------------------

# Ensemble Comparison

  Algorithm           Main Idea                    Major Strength
  ------------------- ---------------------------- -----------------------
  Decision Tree       Single tree                  Simple
  Random Forest       Bagging                      Stable
  Extra Trees         More randomness              Lower variance
  AdaBoost            Reweight mistakes            High precision
  Gradient Boosting   Learn residuals              Strong overall
  XGBoost             Optimized GB                 Regularization
  LightGBM            Leaf-wise GB                 Speed & Recall
  CatBoost            Native categorical support   Minimal preprocessing

------------------------------------------------------------------------

# Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC
-   Confusion Matrix

For imbalanced datasets (e.g., churn prediction), **F1 Score and Recall
are often more informative than Accuracy alone.**

------------------------------------------------------------------------

# Model Selection Guidelines

-   Highest Accuracy → Overall correctness.
-   Highest Precision → Reduce false positives.
-   Highest Recall → Detect as many positives as possible.
-   Highest F1 Score → Best balance of Precision and Recall.

------------------------------------------------------------------------

# Hyperparameters to Tune Later

## Random Forest

-   n_estimators
-   max_depth
-   min_samples_split

## AdaBoost

-   n_estimators
-   learning_rate

## Gradient Boosting

-   n_estimators
-   learning_rate
-   max_depth

## XGBoost

-   max_depth
-   learning_rate
-   subsample
-   colsample_bytree
-   reg_alpha
-   reg_lambda

## LightGBM

-   num_leaves
-   learning_rate
-   max_depth

## CatBoost

-   depth
-   learning_rate
-   iterations

------------------------------------------------------------------------

# Interview Questions

1.  What is Ensemble Learning?
2.  Difference between Bagging and Boosting?
3.  Why does Random Forest reduce overfitting?
4.  Why does AdaBoost use Decision Stumps?
5.  What are residuals in Gradient Boosting?
6.  Why is XGBoost popular?
7.  Explain leaf-wise growth in LightGBM.
8.  What is Ordered Boosting in CatBoost?
9.  Which metric is preferred for imbalanced datasets and why?
10. How would you choose a model for deployment?

------------------------------------------------------------------------

# Key Takeaways

-   Ensemble methods usually outperform a single Decision Tree.
-   Bagging reduces variance; Boosting reduces bias.
-   Gradient Boosting learns residual errors.
-   XGBoost adds optimization and regularization.
-   LightGBM uses leaf-wise growth for efficient learning.
-   CatBoost is designed for categorical data.
-   Compare models using business-relevant metrics, not Accuracy alone.
-   Hyperparameter tuning is essential before deployment.
