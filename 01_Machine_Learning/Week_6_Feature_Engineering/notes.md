# Week 6 - Feature Engineering

## Objective

The goal of Feature Engineering is to prepare raw data into a format that improves Machine Learning model performance.

Dataset Used:
- IBM HR Employee Attrition Dataset
- Rows: 1470
- Original Features: 35
- Target Variable: Attrition

---

# Feature Engineering Pipeline

Raw Data
        │
        ▼
Missing Value Handling
        │
        ▼
Duplicate Removal
        │
        ▼
Remove Unnecessary Columns
        │
        ▼
Encoding
        │
        ▼
Scaling
        │
        ▼
Feature Transformation
        │
        ▼
Feature Analysis
        │
        ▼
Feature Selection
        │
        ▼
Dimensionality Reduction (PCA)
        │
        ▼
Machine Learning Models

---

# 1. Missing Value Handling

Purpose:
Handle missing values before model training.

Techniques:
- Mean
- Median
- Mode
- Constant Value

Decision:

Numerical Features
- Mean (normally distributed)
- Median (skewed)

Categorical Features
- Mode

Example

Age
MonthlyIncome

---

# 2. Duplicate Handling

Purpose

Remove repeated observations.

Code

df.drop_duplicates()

Our Dataset

Duplicate Rows = 0

---

# 3. Remove Unnecessary Features

Columns Removed

EmployeeCount
EmployeeNumber
Over18
StandardHours

Reason

- Constant columns
- Unique IDs
- No predictive information

Features

Before : 35

After : 31

---

# 4. Encoding

## Label Encoding

Used For

Binary categorical variables.

Columns

- Attrition
- Gender
- OverTime

Example

Yes → 1

No → 0

---

## One-Hot Encoding

Used For

Nominal categorical variables.

Columns

- BusinessTravel
- Department
- EducationField
- JobRole
- MaritalStatus

Result

Features

31 → 49

---

# 5. Feature Scaling

Purpose

Bring features to a similar scale.

Technique Used

StandardScaler

Formula

z = (x - mean) / std

Scaled Features

- Age
- DailyRate
- DistanceFromHome
- HourlyRate
- MonthlyIncome
- MonthlyRate
- TotalWorkingYears
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager
- NumCompaniesWorked
- PercentSalaryHike
- TrainingTimesLastYear

Why Scale?

Required for

- Logistic Regression
- SVM
- KNN
- PCA
- Neural Networks

Not required for

- Decision Tree
- Random Forest
- XGBoost

---

# 6. Skewness Analysis

Purpose

Detect non-normal distributions.

Rule

Skewness

- Between -0.5 and 0.5 → Approximately Normal
- Between 0.5 and 1 → Moderately Skewed
- Greater than 1 → Highly Skewed

Dataset Observation

MonthlyIncome

Skewness

Before

1.37

---

# 7. Log Transformation

Purpose

Reduce positive skewness.

Formula

log(x + 1)

Applied On

MonthlyIncome

Result

Before

1.37

After

0.28

Decision

Transformation accepted.

---

# 8. Feature Analysis

Performed

- Distribution plots
- Boxplots
- Summary statistics
- Skewness

Purpose

Understand feature behaviour before model training.

---

# 9. Correlation Analysis

Purpose

Measure linear relationship.

Range

-1 to +1

Interpretation

+1

Perfect Positive

0

No Correlation

-1

Perfect Negative

Top Positive Correlations with Attrition

- OverTime
- MaritalStatus_Single
- Sales Representative
- BusinessTravel_Travel_Frequently

Top Negative Correlations

- MonthlyIncome
- TotalWorkingYears
- JobLevel
- Age
- YearsInCurrentRole

Observation

Higher income and more experience generally reduce attrition.

---

# 10. Variance Threshold

Purpose

Remove features with extremely low variance.

Reason

Low variance features contribute little information.

Result

No feature removed because all remaining features had acceptable variance.

---

# 11. SelectKBest

Method

ANOVA F-Test

Purpose

Select statistically important features.

Top Features

- OverTime
- MonthlyIncome
- MaritalStatus_Single
- TotalWorkingYears
- JobLevel
- Age
- YearsInCurrentRole

---

# 12. Random Forest Feature Importance

Purpose

Measure feature importance using impurity reduction.

Top Features

- MonthlyIncome
- Age
- OverTime
- DailyRate
- TotalWorkingYears
- MonthlyRate

Advantages

Captures

- Non-linear relationships
- Feature interactions

---

# 13. Recursive Feature Elimination (RFE)

Estimator

Logistic Regression

Purpose

Recursively remove least important features.

Selected Features

- JobInvolvement
- MonthlyIncome
- OverTime
- BusinessTravel_Travel_Frequently
- Department_Research & Development
- EducationField_Medical
- MaritalStatus_Single

Observation

RFE removes redundant variables.

---

# Comparison of Feature Selection Methods

Correlation

Measures

Linear relationship

---

SelectKBest

Measures

Statistical significance

---

Random Forest

Measures

Importance from tree splits

---

RFE

Measures

Contribution after recursive elimination

---

Evidence-Based Decision

A feature repeatedly selected by multiple methods is considered more reliable.

Examples

- MonthlyIncome
- OverTime
- Age
- TotalWorkingYears

---

# 14. Principal Component Analysis (PCA)

Purpose

Reduce dimensionality while preserving variance.

Important

PCA creates new features.

It does NOT select original features.

Workflow

Original Features

↓

Principal Components

Example

44 Features

↓

10 Components

---

Important Rule

Always perform PCA after scaling.

Reason

PCA is variance-based.

Large-scale features dominate PCA if scaling is skipped.

---

Our Observation

Without proper scaling

PC1 explained

99.67%

Meaning

One feature dominated the variance.

Conclusion

Scale all numerical features before PCA.

---

When to Use PCA

Recommended

- Logistic Regression
- KNN
- SVM
- Neural Networks

Usually Not Needed

- Decision Trees
- Random Forest
- XGBoost

---

# Industry Feature Engineering Pipeline

Raw Data

↓

EDA

↓

Missing Value Handling

↓

Duplicate Removal

↓

Remove Unnecessary Columns

↓

Encoding

↓

Train-Test Split

↓

Scaling

↓

Feature Transformation

↓

Feature Selection

↓

(Optional) PCA

↓

Machine Learning Model

---

# Best Practices Learned

✓ Understand the data before transforming it.

✓ Do not transform every skewed feature blindly.

✓ Use evidence before applying transformations.

✓ Scale data before PCA.

✓ Remove IDs and constant columns.

✓ Compare multiple feature selection techniques.

✓ Different ML algorithms require different preprocessing.

✓ Feature Engineering should be reproducible using pipelines.

---

# Week 6 Summary

Topics Covered

✓ Missing Values

✓ Duplicate Handling

✓ Remove Unnecessary Features

✓ Label Encoding

✓ One-Hot Encoding

✓ Standard Scaling

✓ Skewness Analysis

✓ Log Transformation

✓ Feature Analysis

✓ Correlation Analysis

✓ Variance Threshold

✓ SelectKBest

✓ Random Forest Feature Importance

✓ Recursive Feature Elimination (RFE)

✓ Principal Component Analysis (PCA)

Status

Week 6 Completed Successfully

Next Week

Week 7 - Machine Learning Algorithms

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)
