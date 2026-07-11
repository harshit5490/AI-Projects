"""
=========================================
Week 6 Capstone Project
Statistics Analyzer
AI Engineer Bootcamp
=========================================
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Generate Dataset
# -------------------------------

# Generate 1000 random values
# Mean = 50
# Standard Deviation = 10

data = np.random.normal(loc=50, scale=10, size=1000)

# -------------------------------
# Step 2: Calculate Statistics
# -------------------------------

mean = np.mean(data)
variance = np.var(data)
std_dev = np.std(data)
minimum = np.min(data)
maximum = np.max(data)

# -------------------------------
# Step 3: Print Statistics
# -------------------------------

print("=" * 50)
print("STATISTICS ANALYZER")
print("=" * 50)

print(f"Mean                : {mean:.2f}")
print(f"Variance            : {variance:.2f}")
print(f"Standard Deviation  : {std_dev:.2f}")
print(f"Minimum Value       : {minimum:.2f}")
print(f"Maximum Value       : {maximum:.2f}")

# -------------------------------
# Step 4: Plot Histogram
# -------------------------------

plt.figure(figsize=(10, 6))

plt.hist(data, bins=30)

# Draw Mean Line
plt.axvline(
    mean,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean:.2f}"
)

plt.title("Normal Distribution (1000 Samples)")
plt.xlabel("Values")
plt.ylabel("Frequency")

plt.legend()

plt.show()

# -------------------------------
# Step 5: Conclusion
# -------------------------------

print("\nConclusion")
print("-" * 50)

print(f"""
The generated dataset approximately follows a Normal Distribution.

Mean:
The average value of the dataset is {mean:.2f}.

Variance:
The variance is {variance:.2f}, which measures how spread out the data is.

Standard Deviation:
The standard deviation is {std_dev:.2f}, meaning that most values lie
approximately {std_dev:.2f} units above or below the mean.

Since the histogram forms a bell-shaped curve and the mean lies near
the center, the dataset resembles a Normal Distribution.
""")