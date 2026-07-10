import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
employees = pd.read_csv(BASE_DIR/"datasets"/"employees.csv")
salaries = pd.read_csv(BASE_DIR/"datasets"/"salaries.csv")
departments = pd.read_csv(BASE_DIR/"datasets"/"departments.csv")

merged_df = pd.merge(employees,salaries,on="EmployeeID")
merged_df = pd.merge(merged_df,departments,on="Department")

merged_df["Bonus"] = merged_df["Salary"]* 0.10
merged_df["Tax"] = merged_df["Salary"]*0.05
merged_df["Net Salary"] = merged_df["Salary"] + merged_df["Bonus"] - merged_df["Tax"]
merged_df["Salary Category"] = merged_df["Salary"].apply(
    lambda Salary:
    "High" if Salary >= 80000 else "Normal"
)
# print(merged_df)
print(f"\nTotal Employee" ,len(merged_df))
print("\nAverage Salary", merged_df["Salary"].mean())
print("\nAverage Age",merged_df["Age"].mean())
print("\nHighest Salary",merged_df["Salary"].max())
print("\nLowest Salary",merged_df["Salary"].min())

print("\nDepartment Wise Average Salary")
print(merged_df.groupby("Department")["Salary"].mean())
print("\nDepartment Wise Employee Count")
print(merged_df.groupby("Department")["EmployeeID"].count())
print("\nHighest Paying Department" , merged_df.groupby("Department")["Salary"].mean().idxmax())
print("\nLowest Paying Department" , merged_df.groupby("Department")["Salary"].mean().idxmin())

department_report = merged_df.groupby("Department").agg(
    Average_Salary=("Salary", "mean"),
    Highest_Salary=("Salary", "max"),
    Lowest_Salary=("Salary", "min"),
    Employee_Count=("EmployeeID", "count"),
    Average_Age=("Age", "mean"),
    Average_Experience=("Experience", "mean")
)
print(department_report)

encoded = pd.get_dummies(merged_df,columns=["Department"])
# print(encoded)
encoded.to_csv(BASE_DIR/"output"/"cleaned_data.csv")
department_report.to_csv(BASE_DIR/"output"/"department_report.csv")