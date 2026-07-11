Statistics

Statistics is the science of collecting, organizing, analyzing, interpreting, and presenting data.

It helps us understand data before building Machine Learning models.

Types of Statistics
1. Descriptive Statistics

Summarizes data.

Examples:

Mean
Median
Mode
Standard Deviation
Variance
2. Inferential Statistics

Uses a sample to make conclusions about a population.

Examples:

Hypothesis Testing
Confidence Intervals
Probability
Measures of Central Tendency

Used to find the center of the dataset.

Mean (Average)
Formula
Mean=
n
∑x
	​

Python
np.mean(data)
Use
Normally distributed data
Missing value imputation (without outliers)
Sensitive to Outliers?

✅ Yes

Median

Middle value after sorting.

Python
np.median(data)
Use
Skewed data
Data with outliers
Salary, income, house prices
Sensitive to Outliers?

❌ No

Mode

Most frequently occurring value.

Python
pd.Series(data).mode()
Use

Categorical data

Examples:

Most common city
Most common product
Most common color
Mean vs Median vs Mode
Measure	Best Use	Outlier Sensitive
Mean	Normal Distribution	✅ Yes
Median	Skewed Data	❌ No
Mode	Categorical Data	❌ No
Measures of Dispersion

Shows how spread out data is.

Range
Formula
Maximum − Minimum
Python
max(data)-min(data)
Outlier Sensitive?

✅ Yes

Variance

Average squared distance from the mean.

Formula
σ
2
=
N
∑(x−μ)
2
	​

Python
np.var(data)

Sample Variance

np.var(data,ddof=1)
Standard Deviation

Square root of variance.

Formula
σ=
Variance
	​

Python
np.std(data)
Interpretation

Small SD

↓

Data is close together.

Large SD

↓

Data is widely spread.

Interquartile Range (IQR)

Measures spread of the middle 50% of data.

Formula
IQR = Q3 - Q1
Python
q1=np.percentile(data,25)
q3=np.percentile(data,75)

iqr=q3-q1
Best Use

Outlier Detection

Quartiles

Divide data into 4 equal parts.

Quartile	Meaning
Q1	25%
Q2	Median (50%)
Q3	75%
Percentiles

Divide data into 100 equal parts.

Example

90th Percentile

↓

90% of observations lie below this value.

Python

np.percentile(data,90)
Five Number Summary
Minimum
Q1
Median
Q3
Maximum

Python

df.describe()
Box Plot Components
Min ----- Q1 ===== Median ===== Q3 ----- Max

Useful for

Distribution
Outlier Detection
IQR
Outlier Detection using IQR

Lower Limit

Q1 - 1.5 × IQR

Upper Limit

Q3 + 1.5 × IQR

Values outside these limits are considered potential outliers.

Population vs Sample
Population	Sample
Entire dataset	Small subset
Divide by N	Divide by n−1
Variance vs Standard Deviation
Variance	Standard Deviation
Squared units	Original units
Harder to interpret	Easy to interpret
Range vs IQR
Range	IQR
Max − Min	Q3 − Q1
Uses only two values	Uses middle 50%
Sensitive to outliers	Robust to outliers
Important NumPy Functions
np.mean()

np.median()

np.var()

np.std()

np.percentile()

np.min()

np.max()
Important Pandas Functions
df.describe()

df.mean()

df.median()

df.mode()

df.var()

df.std()
Machine Learning Applications
Mean
Missing value imputation
Average calculations
Median
Skewed data
Salary prediction
House price datasets
Standard Deviation
StandardScaler
Feature Scaling
IQR
Outlier Detection
Data Cleaning
Percentiles
Feature Engineering
Clipping extreme values
EDA
Interview Questions (Must Remember)
Q1 Why is mean affected by outliers?

Because every value contributes to the total sum.

Q2 Why is median robust to outliers?

Because it depends on the middle position, not the magnitude of extreme values.

Q3 Why do we square deviations in variance?

To prevent positive and negative deviations from canceling each other out.

Q4 Why is Standard Deviation preferred over Variance?

Because it is expressed in the same units as the original data.

Q5 Why do we use IQR?

To detect outliers while ignoring extreme values.

Q6 What does the 75th percentile mean?

75% of the observations are less than or equal to that value.

Q7 What is the Five-Number Summary?

Minimum, Q1, Median, Q3, Maximum.

Q8 Why do we divide by (n−1) for sample variance?

To obtain an unbiased estimate of the population variance (Bessel's correction).

Formula Cheat Sheet
Mean = Σx / n

Range = Max - Min

Variance = Σ(x-μ)² / N

Sample Variance = Σ(x-x̄)² / (n-1)

Standard Deviation = √Variance

IQR = Q3 - Q1

Lower Fence = Q1 - 1.5 × IQR

Upper Fence = Q3 + 1.5 × IQR
Key Takeaways
Mean → Average value (sensitive to outliers)
Median → Middle value (best for skewed data)
Mode → Most frequent value (categorical data)
Range → Simplest measure of spread
Variance → Average squared deviation from the mean
Standard Deviation → Spread in the original units
IQR → Spread of the middle 50%; useful for outlier detection
Percentiles → Show the relative position of a value within a dataset
Five-Number Summary → Minimum, Q1, Median, Q3, Maximum