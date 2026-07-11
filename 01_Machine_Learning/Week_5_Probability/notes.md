# 📘 Week 6: Probability

## 🎯 Learning Outcome

Understand the fundamentals of probability, conditional probability, Bayes' theorem, probability distributions, and statistical measures used in Machine Learning.

---

# 1. Probability

### Formula

P(A) = Number of Favorable Outcomes / Total Number of Outcomes

### Properties

- 0 ≤ P(A) ≤ 1
- P(Impossible Event) = 0
- P(Certain Event) = 1

---

# 2. Sample Space & Events

### Sample Space (S)

Set of all possible outcomes.

Example:

Rolling a die

S = {1,2,3,4,5,6}

### Event

Subset of the sample space.

Example:

Even numbers = {2,4,6}

---

# 3. Conditional Probability

Probability of event A occurring given that event B has already occurred.

### Formula

P(A|B) = P(A ∩ B) / P(B)

### Example

Probability of a student playing cricket given the student is a boy.

---

# 4. Independent vs Dependent Events

## Independent Events

One event does not affect another.

Examples:
- Coin tosses
- Rolling two dice

Formula:

P(A ∩ B) = P(A) × P(B)

---

## Dependent Events

One event affects another.

Examples:
- Drawing cards without replacement
- Drawing balls from a bag without replacement

Formula:

P(A ∩ B) = P(A) × P(B|A)

---

# 5. Bayes' Theorem

Updates the probability after observing new evidence.

### Formula

P(A|B) = [P(B|A) × P(A)] / P(B)

Where:

- Prior = P(A)
- Likelihood = P(B|A)
- Evidence = P(B)
- Posterior = P(A|B)

### Applications

- Spam Detection
- Medical Diagnosis
- Sentiment Analysis
- Recommendation Systems

---

# 6. Random Variable

A variable whose value depends on a random experiment.

## Discrete Random Variable

Countable values.

Examples:
- Die roll
- Number of students
- Number of emails

---

## Continuous Random Variable

Infinite possible values.

Examples:
- Height
- Weight
- Temperature
- Salary

---

# 7. Probability Distribution

Describes how probabilities are distributed over values.

---

## PMF (Probability Mass Function)

Used for Discrete Random Variables.

Example:
- Probability of rolling a 4.

---

## PDF (Probability Density Function)

Used for Continuous Random Variables.

Probability is calculated over intervals.

Example:
- Height between 170 cm and 175 cm.

---

## CDF (Cumulative Distribution Function)

Probability that X ≤ x.

Example:
- Probability of scoring 80 or below.

---

# 8. Common Probability Distributions

## Bernoulli Distribution

Two outcomes.

Examples:
- Pass / Fail
- Yes / No
- Heads / Tails

Applications:
- Logistic Regression
- Binary Classification

---

## Binomial Distribution

Number of successes in n independent Bernoulli trials.

Example:
- Number of heads in 10 coin tosses.

Applications:
- A/B Testing
- Click Prediction

---

## Uniform Distribution

Every value has equal probability.

Example:
- Fair die
- Random numbers between 0 and 1

Applications:
- Random Sampling
- Simulations

---

## Normal (Gaussian) Distribution ⭐⭐⭐⭐⭐

Bell-shaped distribution.

Properties:
- Bell-shaped
- Symmetric
- Mean = Median = Mode
- Area under curve = 1
- Defined by Mean (μ) and Standard Deviation (σ)

### 68–95–99.7 Rule

- 68% within ±1σ
- 95% within ±2σ
- 99.7% within ±3σ

Applications:
- Gaussian Naive Bayes
- Standardization
- Outlier Detection
- Feature Engineering

---

## Poisson Distribution

Models number of events in a fixed interval.

Examples:
- Customers per hour
- Calls per minute
- Website requests

Applications:
- Queueing Systems
- Traffic Prediction

---

# 9. Expected Value (Mean)

Average value of the data.

### Formula

Mean = Σx / n

Python:

np.mean(data)

Applications:
- Missing Value Imputation
- Data Analysis

---

# 10. Variance

Measures how spread out data is.

### Formula

Variance = Σ(x − μ)² / N

Python:

np.var(data)

---

# 11. Standard Deviation

Square root of Variance.

### Formula

SD = √Variance

Python:

np.std(data)

Applications:
- Standardization
- Outlier Detection
- Normal Distribution

---

# Relationship

Mean
↓
Center of Data

Variance
↓
Spread²

Standard Deviation
↓
Spread (Original Units)

---

# Distribution Comparison

| Distribution | Data Type | Example |
|--------------|-----------|---------|
| Bernoulli | Discrete | Pass/Fail |
| Binomial | Discrete | Heads in 10 Tosses |
| Uniform | Both | Fair Die |
| Normal | Continuous | Height |
| Poisson | Discrete | Customers per Hour |

---

# PMF vs PDF vs CDF

| Function | Used For | Meaning |
|----------|----------|---------|
| PMF | Discrete | Exact Probability |
| PDF | Continuous | Density |
| CDF | Both | Probability up to x |

---

# Important NumPy Functions

```python
np.mean()
np.var()
np.std()

np.random.normal()
np.random.binomial()
np.random.uniform()
np.random.poisson()
```

---

# Important Probability Formulas

### Probability

P(A) = Favorable Outcomes / Total Outcomes

### Conditional Probability

P(A|B) = P(A ∩ B) / P(B)

### Independent Events

P(A ∩ B) = P(A) × P(B)

### Dependent Events

P(A ∩ B) = P(A) × P(B|A)

### Bayes' Theorem

P(A|B) = [P(B|A) × P(A)] / P(B)

### Mean

Σx / n

### Variance

Σ(x − μ)² / N

### Standard Deviation

√Variance

---

# Machine Learning Connections

- Probability → Uncertainty Modeling
- Conditional Probability → Naive Bayes
- Bayes' Theorem → Spam Detection, Medical Diagnosis
- Normal Distribution → Gaussian Naive Bayes, Standardization
- Mean → Missing Value Imputation
- Variance → Feature Selection, PCA
- Standard Deviation → Z-score, Outlier Detection
- Poisson Distribution → Event Prediction
- Uniform Distribution → Random Sampling

---

# Interview Questions

1. What is Probability?
2. Difference between Independent and Dependent Events?
3. Explain Conditional Probability.
4. State Bayes' Theorem.
5. What is Prior and Posterior Probability?
6. Difference between PMF, PDF and CDF?
7. Difference between Bernoulli and Binomial Distribution?
8. Why is Normal Distribution important?
9. Explain the 68–95–99.7 Rule.
10. Difference between Variance and Standard Deviation?
11. Why do we square differences in Variance?
12. Can a PDF value be greater than 1?
13. Why is Naive Bayes called "Naive"?
14. Where is Standard Deviation used in Machine Learning?

---

# Week 6 Summary

✅ Probability

✅ Sample Space & Events

✅ Conditional Probability

✅ Independent & Dependent Events

✅ Bayes' Theorem

✅ Random Variables

✅ PMF, PDF & CDF

✅ Bernoulli Distribution

✅ Binomial Distribution

✅ Uniform Distribution

✅ Normal Distribution

✅ Poisson Distribution

✅ Mean

✅ Variance

✅ Standard Deviation

🎯 Week 6 Focus:
Build a strong mathematical foundation for Statistics, Machine Learning, Deep Learning, and AI.