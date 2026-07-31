# Week 14 -- Model Optimization & Explainability Notes

## Topics Covered

1.  Cross Validation
2.  K-Fold Cross Validation
3.  Stratified K-Fold Cross Validation
4.  GridSearchCV
5.  GridSearchCV (`cv_results_`)
6.  RandomizedSearchCV
7.  Bias
8.  Variance
9.  Underfitting
10. Overfitting
11. Bias--Variance Tradeoff
12. Learning Curves
13. Validation Curves
14. Feature Importance
15. SHAP (Theory)
16. Model Saving with joblib
17. Model Saving with pickle
18. Saving Pipelines
19. Deployment Best Practices
20. Interview Questions & Key Takeaways

------------------------------------------------------------------------

## Cross Validation

-   Reliable model evaluation using multiple folds.
-   Every sample becomes validation data exactly once.
-   Common choices: 5-fold and 10-fold.

## K-Fold

-   Split into K folds.
-   Train on K−1 folds and validate on the remaining fold.
-   Repeat K times and average the score.

## Stratified K-Fold

-   Preserves class distribution in every fold.
-   Preferred for classification problems.

## GridSearchCV

-   Exhaustive hyperparameter search.
-   Important attributes:
    -   best_params\_
    -   best_score\_
    -   best_estimator\_
    -   cv_results\_

## RandomizedSearchCV

-   Randomly samples parameter combinations.
-   Faster than GridSearchCV.
-   Main parameter: n_iter.

## Bias

-   Model too simple.
-   Causes underfitting.

## Variance

-   Model too complex.
-   Causes overfitting.

## Bias--Variance Tradeoff

-   Higher complexity → lower bias, higher variance.
-   Goal: best validation performance.

## Learning Curves

-   X-axis: training examples.
-   Y-axis: training and validation score.
-   Used to decide whether more data will help.

## Validation Curves

-   Changes one hyperparameter.
-   Helps choose values such as max_depth or learning_rate.

## Feature Importance

-   Global importance ranking.
-   Supported by tree-based models.
-   Does not explain direction of impact.

## SHAP

-   SHapley Additive exPlanations.
-   Explains both global and local predictions.
-   Practical skipped due to current Python 3.14 / SHAP compatibility
    issue.

## Model Saving

### joblib

``` python
import joblib
joblib.dump(model,"model.joblib")
model=joblib.load("model.joblib")
```

### pickle

``` python
import pickle
pickle.dump(model,file)
model=pickle.load(file)
```

## Pipeline Saving

Save preprocessing and model together for deployment.

## Interview Focus

-   Cross Validation
-   K-Fold vs Stratified K-Fold
-   GridSearchCV vs RandomizedSearchCV
-   Bias vs Variance
-   Learning Curve vs Validation Curve
-   Feature Importance vs SHAP
-   joblib vs pickle

## Key Takeaways

-   Evaluate with Cross Validation.
-   Tune with RandomizedSearchCV and GridSearchCV.
-   Diagnose with Learning & Validation Curves.
-   Explain with Feature Importance and SHAP.
-   Save models and pipelines for production.
