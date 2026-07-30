# Week 12 - Unsupervised Learning

# Table of Contents

1. Introduction to Unsupervised Learning
2. Supervised vs Unsupervised Learning
3. Types of Unsupervised Learning
4. Clustering
5. Distance Metrics
6. K-Means Clustering
7. Working of K-Means
8. Important Parameters
9. K-Means Example
10. Advantages & Disadvantages
11. Interview Questions

---

# 1. Introduction to Unsupervised Learning

## What is Machine Learning?

Machine Learning is divided into three major categories.

```
Machine Learning
│
├── Supervised Learning
│
├── Unsupervised Learning
│
└── Reinforcement Learning
```

---

## What is Unsupervised Learning?

Unsupervised Learning is a Machine Learning technique where the model learns patterns from **unlabeled data**.

Unlike supervised learning, there is **no target column (Y)**.

The algorithm automatically discovers hidden structures and relationships within the dataset.

Example Dataset

| Age | Income | Spending Score |
|-----|--------|----------------|
|22|35|60|
|45|85|20|
|28|75|82|
|52|40|32|

Notice there is **no output column**.

The algorithm itself tries to identify groups.

---

## Real-Life Examples

### Customer Segmentation

Group customers according to purchasing behavior.

---

### News Clustering

Group similar news articles together.

---

### Image Segmentation

Separate different objects inside an image.

---

### Fraud Detection

Detect unusual transactions.

---

### Recommendation Systems

Group users with similar interests.

Examples

- Netflix
- Amazon
- Spotify
- YouTube

---

# 2. Supervised vs Unsupervised Learning

| Supervised Learning | Unsupervised Learning |
|---------------------|-----------------------|
| Uses labeled data | Uses unlabeled data |
| Predicts output | Finds hidden patterns |
| Has target variable | No target variable |
| Regression | Clustering |
| Classification | Dimensionality Reduction |

---

## Example

### Supervised

Dataset

| Area | Bedrooms | Price |
|------|----------|-------|
|1200|2|50L|
|1500|3|70L|

Target

Price

The model learns

Area + Bedrooms

↓

Price

---

### Unsupervised

Dataset

| Area | Bedrooms |
|------|-----------|
|1200|2|
|1500|3|

There is no target.

The algorithm groups similar houses.

---

# 3. Types of Unsupervised Learning

There are mainly two categories.

```
Unsupervised Learning

│

├── Clustering

│

└── Dimensionality Reduction
```

---

## A. Clustering

Purpose

Group similar observations together.

Algorithms

- K-Means
- Hierarchical Clustering
- DBSCAN

Applications

- Customer Segmentation
- Market Basket Analysis
- Fraud Detection
- Social Network Analysis

---

## B. Dimensionality Reduction

Purpose

Reduce the number of features while preserving useful information.

Algorithms

- PCA
- t-SNE

Applications

- Faster Training
- Data Visualization
- Noise Reduction
- Feature Compression

---

# 4. Clustering

## What is Clustering?

Clustering is the process of grouping similar observations together.

Goal

Customers within the same cluster

↓

Should be similar.

Customers in different clusters

↓

Should be different.

---

Example

Suppose we have customers.

```
Customer A

Income = High

Spending = High
```

```
Customer B

Income = High

Spending = High
```

These two customers should belong to the same cluster.

---

Another customer

```
Income = Low

Spending = Low
```

Should belong to another cluster.

---

## Good Clustering

```
● ● ●

● ●


          ▲

▲ ▲ ▲

▲ ▲
```

Clusters are clearly separated.

---

## Bad Clustering

```
● ▲ ● ▲

▲ ● ▲ ●

● ▲ ● ▲
```

Everything overlaps.

---

# Applications of Clustering

## Banking

Customer Segmentation

---

## Healthcare

Disease Grouping

---

## Marketing

Targeted Advertisement

---

## E-commerce

Product Recommendation

---

## Cyber Security

Intrusion Detection

---

# 5. Distance Metrics

Most clustering algorithms measure similarity using distance.

Smaller distance

↓

More similar.

---

## Euclidean Distance

Most common.

Formula

```
d = √((x₂-x₁)² + (y₂-y₁)²)
```

Example

Point A

(2,3)

Point B

(5,7)

Distance

```
√((5−2)²+(7−3)²)

= √(9+16)

= √25

=5
```

---

## Manhattan Distance

Formula

```
|x₂-x₁| + |y₂-y₁|
```

Example

```
|5−2| + |7−3|

=3+4

=7
```

---

## Why K-Means Uses Euclidean Distance?

K-Means minimizes the distance between observations and centroids.

Euclidean Distance gives the shortest straight-line distance.

---

# 6. K-Means Clustering

K-Means is one of the most popular clustering algorithms.

Goal

Partition the dataset into

```
K
```

clusters.

Each cluster has one centroid.

---

## What is K?

K

=

Number of clusters.

Example

```
K=3
```

means

```
Cluster 1

Cluster 2

Cluster 3
```

---

## What is Centroid?

Centroid is the average position of all observations belonging to a cluster.

Think of it as the "center" of a cluster.

Example

```
● ● ●

● X ●

● ● ●
```

X

↓

Centroid

---

# 7. Working of K-Means

The algorithm follows these steps.

```
Choose K

↓

Randomly Initialize Centroids

↓

Assign Every Point

↓

Update Centroids

↓

Repeat Until Convergence
```

---

## Step 1

Choose

```
K
```

Suppose

```
K=2
```

---

## Step 2

Randomly initialize two centroids.

```
X


             X
```

---

## Step 3

Assign every customer to the nearest centroid.

```
● ● ●

X


                 ▲ ▲ ▲

                 X
```

---

## Step 4

Calculate new centroids.

Centroid moves to the average location.

---

## Step 5

Repeat

Assign

↓

Update

↓

Assign

↓

Update

Until centroids stop moving.

This is called

**Convergence**

---

# Why Does K-Means Stop?

The algorithm stops when

- Centroids do not change
- Maximum iterations reached

---

# 8. Important Parameters

## n_clusters

Number of clusters.

Example

```python
KMeans(n_clusters=5)
```

---

## random_state

Makes results reproducible.

```python
random_state=42
```

Without this

Every run may produce different clusters.

---

## n_init

Number of random centroid initializations.

```python
n_init=10
```

The algorithm tries different starting points and chooses the best solution.

Higher

↓

Better clustering

↓

Slightly more computation.

---

## max_iter

Maximum number of iterations.

Default

```
300
```

Increase if convergence warnings occur.

Example

```python
max_iter=1000
```

---

# 9. K-Means Implementation

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)
```

---

Predict clusters

```python
labels = kmeans.predict(X_scaled)
```

or

```python
labels = kmeans.fit_predict(X_scaled)
```

---

Cluster Centers

```python
kmeans.cluster_centers_
```

---

WCSS

```python
kmeans.inertia_
```

---

# 10. Advantages of K-Means

✔ Easy to understand

✔ Fast

✔ Works well on large datasets

✔ Easy to implement

✔ Scalable

---

# Disadvantages

✖ Need to choose K

✖ Sensitive to Outliers

✖ Sensitive to Feature Scaling

✖ Assumes spherical clusters

✖ Different initialization may produce different results

---

# Common Mistakes

❌ Not scaling data

❌ Choosing random K

❌ Ignoring outliers

❌ Using categorical features directly

❌ Assuming highest K always gives best clustering

---

# Best Practices

✔ Handle missing values

✔ Encode categorical features

✔ Standardize numerical features

✔ Use Elbow Method

✔ Verify using Silhouette Score

✔ Interpret clusters using business understanding

---

# Interview Questions

## What is Unsupervised Learning?

Learning patterns from unlabeled data.

---

## What is Clustering?

Grouping similar observations together.

---

## What is K-Means?

A centroid-based clustering algorithm that partitions data into K clusters.

---

## Why is Feature Scaling important?

K-Means uses Euclidean Distance.

Features with larger scales dominate the distance calculation.

Scaling gives equal importance to every feature.

---

## What is a Centroid?

The average point of all observations inside a cluster.

---

## What does random_state do?

Makes the clustering reproducible.

---

## Why use n_init?

To try multiple random centroid initializations and keep the best clustering.

---

## What is Convergence?

When centroids stop changing their positions.

---

# Part 1 Summary

Topics Covered

✅ Introduction to Unsupervised Learning

✅ Supervised vs Unsupervised Learning

✅ Types of Unsupervised Learning

✅ Clustering

✅ Distance Metrics

✅ K-Means Clustering

✅ Working of K-Means

✅ Important Parameters

✅ Advantages & Disadvantages

✅ Interview Questions

---

# Part 2 - Choosing the Optimal Number of Clusters & Hierarchical Clustering

# Table of Contents

11. WCSS (Within Cluster Sum of Squares)
12. Inertia
13. Choosing the Best K
14. Elbow Method
15. Silhouette Score
16. Hierarchical Clustering
17. Dendrogram
18. Linkage Methods
19. Agglomerative Clustering
20. Advantages & Disadvantages
21. Interview Questions

---

# 11. WCSS (Within Cluster Sum of Squares)

## What is WCSS?

WCSS stands for

**Within Cluster Sum of Squares**

It measures how compact a cluster is.

It calculates the squared distance between every observation and its cluster centroid.

Smaller WCSS

↓

Better clustering.

---

## Formula

```
           n
WCSS = Σ (xi − c)²
          i=1
```

Where

xi

=

Data Point

c

=

Cluster Centroid

---

## Understanding WCSS

Suppose

```
Customer

↓

Centroid
```

Distance

```
2
```

Squared Distance

```
2² = 4
```

Another Customer

↓

Distance

```
3
```

Squared Distance

```
9
```

Total

```
WCSS = 4 + 9 = 13
```

---

## Interpretation

Large WCSS

↓

Customers are far from centroid

↓

Poor clustering

---

Small WCSS

↓

Customers are close to centroid

↓

Good clustering

---

# 12. Inertia

In Scikit-Learn,

WCSS is called

```
Inertia
```

Retrieve it using

```python
kmeans.inertia_
```

Example

```python
model = KMeans(n_clusters=5)

model.fit(X_scaled)

print(model.inertia_)
```

Example Output

```
331.31
```

This is the WCSS for

```
K = 5
```

---

# Important Note

Increasing K

↓

Always decreases WCSS

Example

| K | WCSS |
|---|------|
|1|800|
|2|590|
|3|480|
|4|390|
|5|331|
|6|276|

This does NOT mean

```
K=100
```

is the best.

We need a better method.

---

# 13. Choosing the Best K

There is no universal formula.

The two most common methods are

```
1. Elbow Method

2. Silhouette Score
```

Both should be used together.

---

# 14. Elbow Method

## Why?

Since WCSS always decreases,

we need a point where adding another cluster gives only a small improvement.

That point is called

```
Elbow
```

---

## Steps

Choose

```
K = 1

↓

Train Model

↓

Calculate WCSS

↓

Repeat for

K = 2

K = 3

...

K = 10

↓

Plot WCSS
```

---

## Python Code

```python
wcss = []

for k in range(1,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)
```

---

## Plot

```python
plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()
```

---

## Example

```
WCSS

|

|

| *

| *

| *

| *

| *

| *

| *

| *

|____________________________

1 2 3 4 5 6 7 8 9 10
```

Notice

Initially

WCSS decreases rapidly.

Later

The improvement becomes very small.

That bending point

↓

Elbow

↓

Best K

---

## Important Observation

The Elbow is NOT always obvious.

Sometimes

```
K=4
```

and

```
K=5
```

look almost identical.

In that case,

use

Silhouette Score.

---

# Advantages of Elbow Method

✔ Easy

✔ Fast

✔ Widely Used

---

# Disadvantages

✖ Sometimes no clear elbow

✖ Subjective

✖ Different people may choose different K

---

# 15. Silhouette Score

## Why?

The Elbow Method is visual.

Silhouette Score is mathematical.

It measures

How well each observation fits its cluster.

---

## Range

```
-1

↓

0

↓

+1
```

---

### Score Near +1

Excellent Clustering

---

### Score Near 0

Clusters overlap

---

### Negative Score

Wrong clustering

Observation probably belongs to another cluster.

---

## Formula

```
      b − a

S = -----------

    max(a,b)
```

Where

a

Average distance to same cluster.

b

Average distance to nearest neighboring cluster.

---

## Python Code

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(
    X_scaled,
    labels
)

print(score)
```

---

## Example

```
K=2

0.25

K=3

0.31

K=4

0.39

K=5

0.44
```

Highest score

↓

Best clustering

---

## Important Observation

Highest score is NOT always chosen.

Business understanding is equally important.

Example

```
K=5

Score

0.61
```

```
K=6

Score

0.612
```

Difference

Very small.

Business may still prefer

```
K=5
```

because it is easier to explain.

---

# Elbow vs Silhouette

| Elbow | Silhouette |
|--------|------------|
| Visual | Mathematical |
| Uses WCSS | Uses Distance |
| Subjective | Objective |
| Faster | Slightly Slower |

Best Practice

Use both.

---

# 16. Hierarchical Clustering

Hierarchical Clustering builds

a hierarchy of clusters.

Unlike K-Means,

it does NOT start with

random centroids.

---

Two Types

```
Hierarchical

│

├── Agglomerative

└── Divisive
```

---

## Agglomerative

Bottom

↓

Top

Every observation starts as

its own cluster.

Then

closest clusters merge together.

---

## Divisive

Top

↓

Bottom

Everything starts in

one cluster.

Then

splits repeatedly.

---

# Agglomerative Clustering

Workflow

```
200 Customers

↓

200 Clusters

↓

100 Clusters

↓

50 Clusters

↓

20 Clusters

↓

10 Clusters

↓

5 Clusters
```

---

## Python

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=5
)

labels = model.fit_predict(X_scaled)
```

---

# 17. Dendrogram

A Dendrogram is a tree diagram.

Used to decide

Number of Clusters.

---

Example

```
|

|      |

|      |

|   |  |

|   |  |

|___|__|_______
```

Higher merge

↓

Less Similar

Lower merge

↓

More Similar

---

Python

```python
from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import linkage

linked = linkage(
    X_scaled,
    method="ward"
)

dendrogram(linked)

plt.show()
```

---

# 18. Linkage Methods

Hierarchical Clustering needs

Linkage.

Linkage decides

How distance between clusters

is calculated.

---

## Single Linkage

Uses

Nearest Point

Cluster A

● ●

Cluster B

▲ ▲

Distance

Nearest

Point

---

## Complete Linkage

Uses

Farthest Point

Produces compact clusters.

---

## Average Linkage

Uses

Average distance.

---

## Ward Linkage

Most commonly used.

Minimizes variance.

Produces compact clusters.

Recommended for

Customer Segmentation.

---

# Choosing Linkage

| Linkage | Best For |
|----------|----------|
| Single | Long Chains |
| Complete | Compact Clusters |
| Average | Balanced Data |
| Ward | Most ML Problems |

---

# Advantages of Hierarchical Clustering

✔ No random initialization

✔ Produces Dendrogram

✔ Easy to understand

✔ Good for small datasets

---

# Disadvantages

✖ Slow on large datasets

✖ Memory intensive

✖ Cannot undo merges

---

# Interview Questions

## What is WCSS?

Within Cluster Sum of Squares.

Measures compactness of clusters.

Lower is better.

---

## What is Inertia?

Scikit-Learn's name for WCSS.

---

## Why does WCSS always decrease?

Because increasing K makes clusters smaller.

Smaller clusters

↓

Smaller distances

↓

Lower WCSS.

---

## Why can't we always choose the largest K?

Because

Too many clusters

↓

Over-segmentation

↓

Poor business interpretation.

---

## What is the Elbow Method?

Method for selecting K using WCSS.

---

## What is Silhouette Score?

Measures clustering quality.

Range

-1

to

+1

Higher

↓

Better.

---

## Difference between K-Means and Hierarchical Clustering?

K-Means

Requires K.

Hierarchical

Produces Dendrogram.

---

## What is a Dendrogram?

Tree diagram showing how clusters merge.

---

## Which Linkage is commonly used?

Ward Linkage.

---

# Part 2 Summary

Topics Covered

✅ WCSS

✅ Inertia

✅ Choosing K

✅ Elbow Method

✅ Silhouette Score

✅ Hierarchical Clustering

✅ Agglomerative Clustering

✅ Dendrogram

✅ Linkage Methods

✅ Interview Questions

---

# Part 3 - DBSCAN & Principal Component Analysis (PCA)

# Table of Contents

22. DBSCAN
23. Density-Based Clustering
24. Core, Border & Noise Points
25. DBSCAN Parameters
26. DBSCAN Algorithm
27. DBSCAN Implementation
28. Advantages & Disadvantages
29. Curse of Dimensionality
30. Principal Component Analysis (PCA)
31. Covariance Matrix
32. Eigenvalues & Eigenvectors (Concept)
33. Principal Components
34. Explained Variance Ratio
35. Choosing Number of Components
36. PCA Workflow
37. PCA Implementation
38. PCA Applications
39. Interview Questions

---

# 22. DBSCAN

## What is DBSCAN?

DBSCAN stands for

Density-Based Spatial Clustering of Applications with Noise

Unlike K-Means,

DBSCAN

✔ Does NOT require K

✔ Detects Outliers

✔ Finds Arbitrary Shaped Clusters

---

## Why DBSCAN?

Imagine this dataset.

```
● ● ● ●

 ● ● ●

                ▲ ▲ ▲

             ▲ ▲ ▲

                  ■
```

K-Means

↓

Forces every point into a cluster.

DBSCAN

↓

Recognizes

■

as

Noise (Outlier)

---

# 23. Density-Based Clustering

DBSCAN groups points

based on

Density.

High Density

↓

Cluster

Low Density

↓

Noise

---

Example

```
● ● ● ●

● ● ● ●

● ● ● ●
```

High Density

↓

Cluster

---

```
        ▲
```

Single Point

↓

Noise

---

# 24. Types of Points

DBSCAN classifies every observation into

1. Core Point

2. Border Point

3. Noise Point

---

## Core Point

A point having

at least

min_samples

neighbors

inside

eps

radius.

Example

```
● ● ●

● X ●

● ● ●
```

X

↓

Core Point

---

## Border Point

Near a Core Point

But

does not satisfy

minimum neighbors.

Example

```
● ● ●

● X

```

X

↓

Border Point

---

## Noise Point

Not connected to any cluster.

Example

```
                 ▲
```

↓

Noise

---

# 25. DBSCAN Parameters

## eps

Maximum neighborhood distance.

Think of it as

Radius.

```
      ●

   ●  X  ●

      ●
```

Circle

↓

eps

---

## min_samples

Minimum neighbors

required

to become

Core Point.

Example

```
min_samples=5
```

Need

5 nearby observations.

---

# Choosing Parameters

Small eps

↓

Many noise points.

Large eps

↓

Clusters merge together.

Small min_samples

↓

Many tiny clusters.

Large min_samples

↓

Many points become noise.

---

# 26. DBSCAN Algorithm

Step 1

Pick one observation.

↓

Step 2

Count neighbors inside eps.

↓

Enough neighbors?

↓

Yes

↓

Create Cluster.

↓

Expand Cluster.

↓

Repeat

Until all points visited.

---

# Python Implementation

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels = dbscan.fit_predict(X_scaled)
```

---

Outliers

```python
print(labels)
```

Example

```
0

0

1

1

-1

-1
```

-1

means

Noise Point

---

# Advantages

✔ No need to choose K

✔ Detects Outliers

✔ Works on arbitrary shapes

✔ Robust

---

# Disadvantages

✖ Choosing eps is difficult

✖ Poor for varying densities

✖ Slower on large datasets

---

# 27. Curse of Dimensionality

Suppose

Dataset

2 Features

Easy to visualize.

```
X

Y
```

---

Now

100 Features

Impossible to visualize.

Problems

✔ More Memory

✔ Slower Training

✔ Higher Overfitting

✔ Hard Visualization

This is called

Curse of Dimensionality.

---

# Why Reduce Dimensions?

Benefits

✔ Faster Training

✔ Less Memory

✔ Better Visualization

✔ Removes Redundant Features

✔ Noise Reduction

---

# 28. Principal Component Analysis (PCA)

## What is PCA?

Principal Component Analysis

is a

Dimensionality Reduction

Technique.

It transforms

Original Features

↓

New Features

called

Principal Components.

---

Important

PCA

does NOT remove information randomly.

It tries to preserve

Maximum Variance.

---

Example

Original

```
Height

Weight

Income

Age
```

↓

PCA

↓

```
PC1

PC2
```

Most information

is preserved.

---

# 29. Covariance Matrix

PCA first computes

Covariance.

Covariance tells

How two variables change together.

Positive Covariance

↓

Increase together.

Negative Covariance

↓

Move opposite.

Zero

↓

No relationship.

---

Example

Height

Weight

Positive

Age

Exam Marks

Almost Zero

---

# 30. Eigenvalues & Eigenvectors

You DO NOT need deep mathematics.

Interview Understanding

is enough.

---

Eigenvector

↓

Direction

---

Eigenvalue

↓

Importance

---

Large Eigenvalue

↓

Important Direction

Small Eigenvalue

↓

Less Important

---

PCA keeps

Largest Eigenvalues.

---

# 31. Principal Components

Principal Components

are

new features.

Properties

✔ Uncorrelated

✔ Maximum Variance

✔ Ordered by Importance

---

PC1

↓

Highest Variance

PC2

↓

Second Highest

PC3

↓

Third Highest

---

# 32. Explained Variance Ratio

Shows

How much information

each Principal Component captures.

Example

```
PC1

65%

PC2

20%

PC3

10%

PC4

5%
```

Total

100%

---

Python

```python
print(
pca.explained_variance_ratio_
)
```

Output

```
0.65

0.20

0.10

0.05
```

---

Interpretation

PC1

captures

65%

of total information.

---

# Choosing Number of Components

Option 1

Choose manually

```python
PCA(
n_components=2
)
```

---

Option 2

Keep

95%

Variance.

```python
PCA(
n_components=0.95
)
```

Automatically selects

required components.

---

# 33. PCA Workflow

```
Original Dataset

↓

StandardScaler

↓

Covariance Matrix

↓

Eigenvalues

↓

Eigenvectors

↓

Sort Components

↓

Choose Top Components

↓

Transform Data
```

---

# Why Scale Before PCA?

PCA

depends on

Variance.

Large scale features

dominate.

Always

Standardize

before PCA.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# PCA Implementation

```python
from sklearn.decomposition import PCA

pca = PCA(
n_components=2
)

X_pca = pca.fit_transform(X_scaled)
```

---

Check Explained Variance

```python
print(
pca.explained_variance_ratio_
)
```

---

# PCA Applications

✔ Visualization

✔ Noise Removal

✔ Compression

✔ Faster Training

✔ Feature Extraction

✔ Reduce Overfitting

---

# Advantages

✔ Faster Models

✔ Less Memory

✔ Removes Correlation

✔ Easy Visualization

---

# Disadvantages

✖ Less Interpretability

✖ Information Loss

✖ Linear Technique

✖ Scaling Required

---

# Interview Questions

## What is DBSCAN?

A density-based clustering algorithm.

---

## Why DBSCAN over K-Means?

It detects

Outliers

and

does not require K.

---

## What are Core Points?

Points having enough neighbors.

---

## What is eps?

Neighborhood radius.

---

## What is min_samples?

Minimum neighbors required to become a Core Point.

---

## What is Curse of Dimensionality?

Problems caused by high-dimensional data such as increased computation, overfitting, and difficult visualization.

---

## What is PCA?

A dimensionality reduction technique that transforms correlated features into uncorrelated principal components while preserving maximum variance.

---

## Why use PCA?

✔ Reduce dimensions

✔ Faster training

✔ Better visualization

✔ Remove redundancy

---

## Why StandardScaler before PCA?

PCA depends on variance.

Without scaling,

large-value features dominate.

---

## What is Explained Variance Ratio?

Percentage of information preserved by each Principal Component.

---

## Difference Between Feature Selection and PCA

Feature Selection

↓

Keeps original features.

PCA

↓

Creates new transformed features.

---

# Part 3 Summary

Topics Covered

✅ DBSCAN

✅ Density-Based Clustering

✅ Core Points

✅ Border Points

✅ Noise Points

✅ eps

✅ min_samples

✅ Curse of Dimensionality

✅ PCA

✅ Covariance Matrix

✅ Eigenvalues

✅ Eigenvectors

✅ Principal Components

✅ Explained Variance Ratio

✅ PCA Implementation

✅ Interview Questions

---

# Part 4 - t-SNE & Customer Segmentation Project

# Table of Contents

40. t-SNE
41. How t-SNE Works
42. t-SNE Parameters
43. PCA vs t-SNE
44. Customer Segmentation Project
45. Project Workflow
46. Data Preprocessing
47. Feature Scaling
48. Choosing K
49. Model Training
50. Cluster Interpretation
51. PCA Visualization
52. Business Recommendations
53. Interview Questions

---

# 40. t-SNE

## What is t-SNE?

t-SNE stands for

t-Distributed Stochastic Neighbor Embedding

It is a **non-linear dimensionality reduction** algorithm.

Main purpose

✔ Visualization

It reduces high-dimensional data into

- 2D
- 3D

for visualization.

---

## Why use t-SNE?

Imagine a dataset with

```
500 Features
```

Humans cannot visualize it.

t-SNE converts

```
500 Features

↓

2 Features
```

making visualization possible.

---

## Characteristics

✔ Excellent visualization

✔ Preserves local relationships

✔ Works well on complex datasets

✔ Non-linear

---

## Limitation

t-SNE is **NOT** used for feature reduction before training ML models.

It is mainly used for visualization.

---

# 41. How t-SNE Works

Unlike PCA,

t-SNE tries to keep

Nearby points

↓

Nearby after transformation.

Example

Original

```
● ● ●

▲ ▲ ▲
```

After t-SNE

```
● ● ●


▲ ▲ ▲
```

The neighborhood relationship is preserved.

---

# 42. Important Parameters

## n_components

Number of dimensions.

```python
TSNE(n_components=2)
```

---

## perplexity

Controls neighborhood size.

Typical values

```
5

↓

50
```

Common choice

```
30
```

---

## random_state

Ensures reproducible results.

```python
random_state=42
```

---

# Implementation

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)
```

---

# 43. PCA vs t-SNE

| PCA | t-SNE |
|------|--------|
| Linear | Non-linear |
| Fast | Slow |
| Feature Reduction | Visualization |
| Preserves Variance | Preserves Neighborhood |
| Deterministic | Slightly Random |
| Can be used before ML | Mostly Visualization |

---

## Which one should we use?

Use PCA

✔ Faster training

✔ Feature reduction

✔ Compression

Use t-SNE

✔ Data visualization

✔ Cluster visualization

✔ High-dimensional datasets

---

# 44. Customer Segmentation Project

Dataset

Mall Customers Dataset

Goal

Segment customers based on purchasing behavior.

---

## Features

- Gender
- Age
- Annual Income
- Spending Score

---

## Business Objective

Identify different customer groups

↓

Provide personalized marketing strategies.

---

# 45. Complete Workflow

```
Load Dataset

↓

EDA

↓

Handle Missing Values

↓

Encode Categorical Features

↓

Feature Scaling

↓

Choose K

↓

Train KMeans

↓

Evaluate

↓

PCA Visualization

↓

Business Interpretation
```

---

# 46. Data Preprocessing

## Step 1

Load Dataset

```python
df = pd.read_csv("Mall_Customers.csv")
```

---

## Step 2

Check Missing Values

```python
df.isnull().sum()
```

---

## Step 3

Encode Gender

```python
encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])
```

---

## Step 4

Remove CustomerID

CustomerID

↓

Unique Identifier

↓

No useful information for clustering.

```python
df.drop(
    "CustomerID",
    axis=1,
    inplace=True
)
```

---

# 47. Feature Scaling

Since K-Means uses

Euclidean Distance

Scaling is compulsory.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)
```

---

# 48. Choosing K

Method 1

Elbow Method

↓

Find bending point.

Method 2

Silhouette Score

↓

Choose highest score.

---

# 49. Model Training

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
```

---

# Cluster Centers

```python
kmeans.cluster_centers_
```

Convert back

```python
scaler.inverse_transform(
    kmeans.cluster_centers_
)
```

---

# 50. Cluster Interpretation

Each centroid represents

Average Customer

Example

| Cluster | Age | Income | Spending |
|---------|----|--------|----------|
|0|32|86|82|

Interpretation

Young

High Income

High Spending

↓

Premium Customer

---

# Example Segments

## Cluster 0

High Income

High Spending

↓

Premium Customers

---

## Cluster 1

High Income

Low Spending

↓

Potential Customers

---

## Cluster 2

Average Income

Average Spending

↓

Regular Customers

---

## Cluster 3

Young

High Spending

↓

Young Active Shoppers

---

## Cluster 4

Older

Lower Spending

↓

Conservative Customers

---

# 51. PCA Visualization

Since our dataset has

4 Features

We reduce it

↓

2 Components

using PCA.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

---

## Scatter Plot

```python
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=clusters
)
```

Each color

↓

One Cluster

---

## Plot Centroids

```python
centroids = pca.transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker="X",
    s=250,
    c="black"
)
```

Black X

↓

Cluster Centroid

---

# Interpretation

Well-separated clusters

↓

Good segmentation

Overlapping clusters

↓

Need better features or different K.

---

# 52. Business Recommendations

## Premium Customers

Characteristics

✔ High Income

✔ High Spending

Recommendation

- VIP Membership
- Luxury Products
- Exclusive Discounts

---

## Potential Customers

Characteristics

✔ High Income

✔ Low Spending

Recommendation

- Personalized Discounts
- Loyalty Programs
- Email Campaigns

---

## Regular Customers

Recommendation

- Seasonal Offers
- Cashback
- Combo Deals

---

## Young Active Shoppers

Recommendation

- Student Discounts
- Flash Sales
- Social Media Marketing

---

## Conservative Customers

Recommendation

- Personalized Service
- Budget-Friendly Products
- Family Packages

---

# 53. Interview Questions

## Why use Customer Segmentation?

To identify different customer groups and improve business decisions.

---

## Why remove CustomerID?

It is only an identifier and contains no useful information.

---

## Why scale before K-Means?

Because K-Means uses Euclidean Distance.

---

## Why use PCA?

To visualize clusters in two dimensions.

---

## Was PCA used to train K-Means?

No.

K-Means was trained on

Scaled Features.

PCA was used only for visualization.

---

## What do centroids represent?

Average customer of a cluster.

---

## Why interpret centroids?

To understand customer behavior and create business strategies.

---

# Part 4 Summary

Topics Covered

✅ t-SNE

✅ PCA vs t-SNE

✅ Customer Segmentation

✅ Data Preprocessing

✅ Feature Scaling

✅ Model Training

✅ Cluster Interpretation

✅ PCA Visualization

✅ Business Recommendations

✅ Interview Questions

---

# Part 5 - Model Saving, Deployment, Best Practices & Interview Preparation

# Table of Contents

54. Model Persistence
55. Joblib
56. Saving K-Means Model
57. Saving StandardScaler
58. Loading Saved Models
59. Predicting New Customers
60. Project Folder Structure
61. Common Mistakes
62. Best Practices
63. Real World Applications
64. Advanced Interview Questions

---

# 54. Model Persistence

## What is Model Persistence?

Model Persistence means

Saving a trained model

↓

Loading it later

without training again.

Instead of

```
Dataset

↓

Train

↓

Predict
```

Every time,

we do

```
Train Once

↓

Save Model

↓

Load Model

↓

Predict
```

This saves

✔ Time

✔ CPU

✔ Memory

---

# Why Save Models?

Suppose

Training Time

```
30 Minutes
```

Without saving

Every execution

↓

30 Minutes

With saving

```
Load Model

↓

2 Seconds
```

---

# 55. Joblib

Scikit-Learn recommends

```
Joblib
```

for saving models.

---

## Why Joblib?

✔ Fast

✔ Efficient

✔ Optimized for NumPy Arrays

✔ Recommended by Scikit-Learn

---

Import

```python
import joblib
```

---

# 56. Saving K-Means Model

```python
joblib.dump(
    kmeans,
    "kmeans_customer_segmentation.pkl"
)
```

Output

```
Model Saved Successfully
```

Generated File

```
kmeans_customer_segmentation.pkl
```

---

# 57. Saving StandardScaler

Very Important

Always save

StandardScaler

because

New data must be scaled

exactly like

training data.

```python
joblib.dump(
    scaler,
    "scaler.pkl"
)
```

Now we have

```
kmeans_customer_segmentation.pkl

scaler.pkl
```

---

# Why Save the Scaler?

Training

```
Original Data

↓

StandardScaler

↓

Scaled Data

↓

KMeans
```

Prediction

```
New Customer

↓

StandardScaler

↓

Scaled Data

↓

KMeans
```

If we don't save the scaler,

new data will be transformed differently,

leading to incorrect predictions.

---

# 58. Loading Saved Models

Load KMeans

```python
loaded_model = joblib.load(
    "kmeans_customer_segmentation.pkl"
)
```

Load Scaler

```python
loaded_scaler = joblib.load(
    "scaler.pkl"
)
```

Now

No retraining required.

---

# 59. Predicting New Customers

Suppose

New Customer

| Gender | Age | Income | Spending |
|--------|-----|--------|----------|
| Male | 28 | 90 | 80 |

---

## Step 1

Create DataFrame

```python
new_customer = pd.DataFrame({

    "Gender":[0],

    "Age":[28],

    "Annual Income (k$)":[90],

    "Spending Score (1-100)":[80]

})
```

---

## Step 2

Scale Data

```python
new_scaled = loaded_scaler.transform(
    new_customer
)
```

---

## Step 3

Predict Cluster

```python
prediction = loaded_model.predict(
    new_scaled
)

print(prediction)
```

Example

```
Cluster = 0
```

---

## Step 4

Interpret Cluster

Suppose

Cluster 0

↓

Premium Customers

Business Decision

Offer

- VIP Membership

- Luxury Products

- Exclusive Discounts

---

# 60. Project Folder Structure

```
Week_12_Unsupervised_Learning/

│

├── assignment.py

├── notes.md

├── Mall_Customers.csv

│

├── kmeans_customer_segmentation.pkl

├── scaler.pkl

│

├── images/

│      ├── elbow_curve.png

│      ├── silhouette_score.png

│      ├── pca_clusters.png

│      ├── centroid_plot.png

│

└── README.md
```

---

# README Should Contain

Project Name

Objective

Dataset

Workflow

Algorithms Used

Results

Business Insights

Future Improvements

---

# Common Mistakes

## Mistake 1

Not Scaling Data

Wrong

```python
kmeans.fit(X)
```

Correct

```python
kmeans.fit(X_scaled)
```

---

## Mistake 2

Choosing Random K

Always use

✔ Elbow Method

✔ Silhouette Score

---

## Mistake 3

Ignoring Outliers

Outliers affect

K-Means

significantly.

---

## Mistake 4

Using CustomerID

CustomerID

↓

Identifier

↓

No useful information.

Remove it.

---

## Mistake 5

Not Saving Scaler

Saving only

KMeans

is incomplete.

---

## Mistake 6

Using PCA Before Understanding Data

PCA should be applied

after

data cleaning

and

feature scaling.

---

## Mistake 7

Using Too Many Clusters

More clusters

≠

Better model.

---

# Best Practices

✔ Handle Missing Values

✔ Remove Duplicate Records

✔ Encode Categorical Variables

✔ Standardize Features

✔ Choose K Scientifically

✔ Validate Using Silhouette Score

✔ Interpret Every Cluster

✔ Save Model

✔ Save Scaler

✔ Document Workflow

---

# Real World Applications

## Banking

Customer Segmentation

---

## Insurance

Risk Profiling

---

## Healthcare

Disease Clustering

---

## Cyber Security

Anomaly Detection

---

## Marketing

Customer Targeting

---

## Retail

Product Recommendation

---

## E-commerce

Purchase Behavior Analysis

---

## Telecom

Customer Churn Analysis

---

## Manufacturing

Machine Failure Detection

---

# Advanced Interview Questions

## What is Model Persistence?

Saving a trained model

to reuse later.

---

## Why Joblib instead of Pickle?

Joblib is

✔ Faster

✔ More Efficient

✔ Optimized for NumPy Arrays

---

## Why Save StandardScaler?

Because new data

must be transformed

using

the same scaling

used during training.

---

## Why remove CustomerID?

Unique Identifier

No predictive value.

---

## Why use PCA only for visualization?

Because

PCA reduces dimensions.

KMeans should be trained

on the complete

scaled feature set

unless dimensionality reduction

is actually required.

---

## Why does KMeans fail on non-spherical clusters?

Because

it assumes

Euclidean distance

around centroids.

---

## When should DBSCAN be preferred?

When

✔ Outliers exist

✔ Clusters have arbitrary shapes

✔ Number of clusters is unknown

---

## Difference Between PCA and Feature Selection

Feature Selection

↓

Keeps original features.

PCA

↓

Creates new transformed features.

---

## What happens if we don't scale data?

Features with larger values

dominate

Euclidean Distance.

---

## Which clustering algorithm is best?

No universal answer.

Depends on

✔ Dataset

✔ Shape

✔ Density

✔ Business Problem

---

## How do you evaluate clustering?

✔ Elbow Method

✔ Silhouette Score

✔ Business Interpretation

---

## Which algorithm detects outliers?

DBSCAN

---

## Which algorithm needs K?

KMeans

Agglomerative (when implemented with n_clusters)

---

## Which algorithm produces Dendrogram?

Hierarchical Clustering

---

## Which dimensionality reduction technique is linear?

PCA

---

## Which dimensionality reduction technique is non-linear?

t-SNE

---

# Part 5 Summary

Topics Covered

✅ Model Persistence

✅ Joblib

✅ Saving Models

✅ Saving Scaler

✅ Loading Models

✅ Predicting New Customers

✅ Project Structure

✅ Common Mistakes

✅ Best Practices

✅ Real World Applications

✅ Advanced Interview Questions

---
# Part 6 - Quick Revision, Cheat Sheet & Final Summary

# Table of Contents

65. Important Formulas
66. Important Scikit-Learn Functions
67. Algorithm Comparison
68. PCA vs t-SNE
69. Quick Revision Cheat Sheet
70. End-to-End Workflow
71. Common Interview Questions
72. Learning Outcomes
73. Week 12 Summary
74. Revision Checklist

---

# 65. Important Formulas

## Euclidean Distance

Used by K-Means

```
             ______________________

d = √ Σ(x₂ − x₁)²
```

Purpose

Measure distance between two observations.

---

## WCSS

Within Cluster Sum of Squares

```
           n

WCSS = Σ (xi − c)²
          i=1
```

Lower

↓

Better clustering.

---

## Silhouette Score

```
      b − a

S = -----------

    max(a,b)
```

Where

a

Average distance to same cluster.

b

Average distance to nearest cluster.

Range

```
-1

↓

0

↓

+1
```

Higher

↓

Better.

---

## Covariance

Measures relationship between variables.

Positive

↓

Increase together.

Negative

↓

Move opposite.

Zero

↓

No relationship.

---

# 66. Important Scikit-Learn Functions

## KMeans

```python
KMeans()
```

---

Train

```python
fit()
```

---

Predict

```python
predict()
```

---

Train + Predict

```python
fit_predict()
```

---

Centroids

```python
cluster_centers_
```

---

WCSS

```python
inertia_
```

---

## Silhouette Score

```python
silhouette_score()
```

---

## PCA

```python
PCA()
```

Transform

```python
fit_transform()
```

---

Explained Variance

```python
explained_variance_ratio_
```

---

## DBSCAN

```python
DBSCAN()
```

---

## Agglomerative Clustering

```python
AgglomerativeClustering()
```

---

## Joblib

Save

```python
joblib.dump()
```

Load

```python
joblib.load()
```

---

# 67. Algorithm Comparison

| Feature | K-Means | Hierarchical | DBSCAN |
|----------|----------|--------------|---------|
| Need K | ✅ Yes | Usually Yes | ❌ No |
| Detect Outliers | ❌ No | ❌ No | ✅ Yes |
| Dendrogram | ❌ | ✅ | ❌ |
| Speed | Fast | Slow | Medium |
| Large Dataset | Excellent | Poor | Good |
| Arbitrary Shape | ❌ | Limited | ✅ |
| Random Initialization | Yes | No | No |

---

# 68. PCA vs t-SNE

| PCA | t-SNE |
|------|--------|
| Linear | Non-linear |
| Fast | Slow |
| Feature Reduction | Visualization |
| Preserves Variance | Preserves Neighborhood |
| Deterministic | Randomized |
| Used Before ML | Mainly Visualization |

---

# K-Means vs DBSCAN

| K-Means | DBSCAN |
|----------|---------|
| Uses Centroids | Uses Density |
| Need K | No K |
| Sensitive to Outliers | Detects Outliers |
| Spherical Clusters | Arbitrary Shapes |
| Fast | Slower |

---

# Supervised vs Unsupervised

| Supervised | Unsupervised |
|-------------|--------------|
| Has Labels | No Labels |
| Predicts Output | Finds Patterns |
| Regression | Clustering |
| Classification | PCA |

---

# 69. Quick Revision Cheat Sheet

## K-Means

Goal

Group similar observations.

Needs

✔ K

Uses

✔ Euclidean Distance

Evaluated by

✔ WCSS

✔ Silhouette Score

---

## Hierarchical

Produces

✔ Dendrogram

No random initialization.

Best for

Small datasets.

---

## DBSCAN

Uses

Density.

Parameters

✔ eps

✔ min_samples

Detects

✔ Noise

✔ Outliers

---

## PCA

Purpose

Reduce dimensions.

Creates

Principal Components.

Preserves

Maximum Variance.

Requires

StandardScaler.

---

## t-SNE

Purpose

Visualization.

Preserves

Local Neighborhood.

Not used

for model training.

---

# 70. End-to-End Workflow

```
Load Dataset

↓

EDA

↓

Missing Values

↓

Encode

↓

Scale

↓

Choose Algorithm

↓

Train Model

↓

Evaluate

↓

Interpret Results

↓

Visualize

↓

Save Model

↓

Deploy
```

---

# Customer Segmentation Workflow

```
Mall Customers Dataset

↓

EDA

↓

Label Encoding

↓

StandardScaler

↓

Elbow Method

↓

Silhouette Score

↓

KMeans

↓

Cluster Assignment

↓

PCA Visualization

↓

Business Insights

↓

Save Model
```

---

# 71. Frequently Asked Interview Questions

## What is Unsupervised Learning?

Learning patterns from unlabeled data.

---

## Why use K-Means?

Simple

Fast

Scalable

Easy to interpret.

---

## Why StandardScaler?

Because K-Means uses Euclidean Distance.

---

## What is WCSS?

Within Cluster Sum of Squares.

Measures compactness.

---

## What is Inertia?

Scikit-Learn's implementation of WCSS.

---

## Why Elbow Method?

To estimate optimal K.

---

## Why Silhouette Score?

To evaluate clustering quality.

---

## What is DBSCAN?

Density-based clustering algorithm.

---

## Why PCA?

Reduce dimensions while preserving maximum variance.

---

## Why save StandardScaler?

New data must be transformed exactly like training data.

---

## Difference between PCA and t-SNE?

PCA

↓

Feature Reduction

t-SNE

↓

Visualization.

---

## What is a Centroid?

Average position of all observations inside a cluster.

---

## Which algorithm detects Outliers?

DBSCAN.

---

## Which algorithm produces a Dendrogram?

Hierarchical Clustering.

---

## Which algorithm needs K?

K-Means.

---

# 72. Learning Outcomes

After completing Week 12,

you can

✔ Explain Unsupervised Learning

✔ Apply K-Means

✔ Choose Optimal K

✔ Use Elbow Method

✔ Use Silhouette Score

✔ Apply Hierarchical Clustering

✔ Interpret Dendrogram

✔ Apply DBSCAN

✔ Detect Outliers

✔ Understand Curse of Dimensionality

✔ Apply PCA

✔ Explain Explained Variance

✔ Compare PCA and t-SNE

✔ Build Customer Segmentation Project

✔ Interpret Business Insights

✔ Save Models

✔ Load Models

✔ Predict New Customers

---

# 73. Week 12 Summary

Topics Covered

✅ Introduction to Unsupervised Learning

✅ Clustering

✅ Distance Metrics

✅ K-Means

✅ WCSS

✅ Elbow Method

✅ Silhouette Score

✅ Hierarchical Clustering

✅ Dendrogram

✅ Linkage Methods

✅ DBSCAN

✅ Curse of Dimensionality

✅ PCA

✅ Covariance

✅ Eigenvalues

✅ Eigenvectors

✅ Explained Variance Ratio

✅ t-SNE

✅ Customer Segmentation

✅ Business Interpretation

✅ Joblib

✅ Model Persistence

✅ Interview Preparation

---

# 74. Revision Checklist

Before moving to Week 13, ensure you can answer:

☐ What is Unsupervised Learning?

☐ Difference between Supervised and Unsupervised Learning?

☐ Explain K-Means step-by-step.

☐ Why is StandardScaler necessary?

☐ What is WCSS?

☐ What is Inertia?

☐ Explain the Elbow Method.

☐ Explain the Silhouette Score.

☐ Difference between K-Means and DBSCAN.

☐ Difference between Hierarchical and K-Means.

☐ Explain Dendrogram.

☐ What are Core, Border, and Noise Points?

☐ Explain eps and min_samples.

☐ What is the Curse of Dimensionality?

☐ Explain PCA.

☐ Why use PCA?

☐ Explain Explained Variance Ratio.

☐ Difference between PCA and t-SNE.

☐ Explain the Customer Segmentation Project.

☐ Why save the model and scaler?

☐ Predict a new customer using the saved model.

---

# 🎉 Congratulations!

You have successfully completed **Week 12 – Unsupervised Learning**.

You are now proficient in:

- Clustering Techniques
- Dimensionality Reduction
- Customer Segmentation
- Model Persistence
- Business Interpretation
- Interview-Level Concepts

This knowledge forms the foundation for recommendation systems, anomaly detection, market segmentation, and many real-world machine learning applications.

The next milestone in your roadmap is:

# Week 13 – Ensemble Learning

Topics include:

- Voting Classifier
- Bagging
- Random Forest
- Extra Trees
- Boosting
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Feature Importance
- Ensemble Learning Project
