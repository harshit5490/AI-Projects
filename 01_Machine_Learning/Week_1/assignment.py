# Q1. Why was NumPy created?

# Q2. How is a Python list stored in memory?

# Q3. What is a pointer?

# Q4. Why are NumPy arrays faster?

# Q5. What is contiguous memory?

import numpy as np

# ==================================================
# Employee Performance Analytics System
# ==================================================

# Dataset
employees = np.array([
    [85, 90, 88, 95, 80],
    [70, 65, 75, 80, 72],
    [92, 88, 95, 90, 94],
    [60, 70, 68, 75, 65],
    [78, 82, 80, 85, 79],
    [95, 96, 94, 98, 97],
    [55, 60, 58, 65, 62],
    [88, 85, 90, 92, 87]
])

subjects = [
    "Coding",
    "Communication",
    "Problem Solving",
    "Attendance",
    "Teamwork"
]

# ==================================================
# Part 1 - Dataset Information
# ==================================================

print("=" * 55)
print("        Employee Performance Report")
print("=" * 55)

print(f"\nNumber of Employees : {employees.shape[0]}")
print(f"Number of Features  : {employees.shape[1]}")
print(f"Dataset Shape       : {employees.shape}")
print(f"Data Type           : {employees.dtype}")

# ==================================================
# Part 2 - Statistics
# ==================================================

print("\n" + "=" * 55)
print("Subject-wise Statistics")
print("=" * 55)

average_scores = np.mean(employees, axis=0)
maximum_scores = np.max(employees, axis=0)
minimum_scores = np.min(employees, axis=0)

for i in range(len(subjects)):
    print(f"\n{subjects[i]}")
    print(f"Average : {average_scores[i]:.2f}")
    print(f"Highest : {maximum_scores[i]}")
    print(f"Lowest  : {minimum_scores[i]}")

# ==================================================
# Part 3 - Coding Score > 85
# ==================================================

print("\n" + "=" * 55)
print("Employees with Coding Score > 85")
print("=" * 55)

high_coders = employees[employees[:, 0] > 85]

print(high_coders)

# ==================================================
# Part 4 - Increase Attendance by 5%
# ==================================================

print("\n" + "=" * 55)
print("Attendance After 5% Increment")
print("=" * 55)

updated_employees = employees.astype(float)

updated_employees[:, 3] *= 1.05

print(updated_employees[:, 3])

# ==================================================
# Part 5 - Overall Performance Score
# ==================================================

weights = np.array([
    0.35,
    0.15,
    0.25,
    0.10,
    0.15
])

performance_scores = employees @ weights
# print(performance_scores)

print("\n" + "=" * 55)
print("Performance Scores")
print("=" * 55)

for i, score in enumerate(performance_scores, start=1):
    print(f"Employee {i} : {score:.2f}")

# ==================================================
# Part 6 - Best and Worst Employee
# ==================================================

best_employee = np.argmax(performance_scores)
worst_employee = np.argmin(performance_scores)

print("\n" + "=" * 55)
print("Overall Analysis")
print("=" * 55)

print(f"Best Employee      : Employee {best_employee + 1}")
print(f"Best Score         : {performance_scores[best_employee]:.2f}")

print(f"\nWorst Employee     : Employee {worst_employee + 1}")
print(f"Worst Score        : {performance_scores[worst_employee]:.2f}")

print(f"\nOverall Average    : {np.mean(performance_scores):.2f}")

# ==================================================
# Final Report
# ==================================================

print("\n" + "=" * 55)
print("Summary Report")
print("=" * 55)

print(f"Total Employees            : {employees.shape[0]}")
print(f"Average Coding             : {average_scores[0]:.2f}")
print(f"Average Communication      : {average_scores[1]:.2f}")
print(f"Average Problem Solving    : {average_scores[2]:.2f}")
print(f"Average Attendance         : {average_scores[3]:.2f}")
print(f"Average Teamwork           : {average_scores[4]:.2f}")

print("-" * 55)

print(f"Top Performer              : Employee {best_employee + 1}")
print(f"Performance Score          : {performance_scores[best_employee]:.2f}")

print("-" * 55)

print(f"Lowest Performer           : Employee {worst_employee + 1}")
print(f"Performance Score          : {performance_scores[worst_employee]:.2f}")

print("-" * 55)

print(f"Overall Average Score      : {np.mean(performance_scores):.2f}")

print("=" * 55)
print("Report Generated Successfully!")
print("=" * 55)