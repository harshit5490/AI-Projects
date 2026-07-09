import pandas as pd
import numpy as np
from pathlib import Path

Base_DIR = Path(__file__).parent

data = {
    "Name":["Harshit","Amit","Rahul"],
    "Age": [24, 26, 22],
    "Salary": [50000, 60000, 45000]
}

df = pd.DataFrame(data)
# print(df)
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.info())
# print(df.describe())

df = pd.read_csv("01_Machine_Learning/Week_2/datasets/employee.csv")
# print(df)
# df.info()
# print(df.head())
# print(df.tail())
# print(df["Salary"].max())
# df.to_csv("01_Machine_Learning/Week_2/datasets/new_employee.csv",index=False)

# print(type(df["Name"]))
# print(type(df[["Name"]]))

# print(df.iloc[1])
# print(df.iloc[0:5])

# print(df.iloc[:,[1,4]])

# print(df.loc[0])
# print(df.loc[:,["Name"]])

# print(df.loc[0,"Name"])

result = df[
    (df["Department"] == "AI") &
    (df["Salary"] > 70000)
]

# print(result)
# print(df[df["Department"] != "HR"])
# print(df[df["Department"].isin(["AI","HR","Finance"])])
# print(df[df["Age"].between(24,28)])

# print(df.loc[df["Salary"]>70000,["Name","Salary"]])

sorted_df = df.sort_values(by="Salary")

# print(sorted_df)

sorted_df = df.sort_values(by="Salary",ascending=False)
# print(sorted_df)

sorted_df = df.sort_values(
    by=["Department", "Salary"],
    ascending=[True, False]
)

# print(sorted_df)
# df.sort_values(by="Salary", inplace=True)
# print(df)
sorted_df = sorted_df.reset_index(drop=True)
# print(sorted_df)

# print(df.groupby("Department")["Salary"].mean())

# print(
#     df.groupby("Department")["Salary"]
#       .agg(["mean", "max", "min", "sum", "count"])
# )

# print(df.groupby("Department")[["Salary","Experience"]].mean())

employees = pd.read_csv(Base_DIR/"datasets"/"employees.csv")
salaries = pd.read_csv(Base_DIR/"datasets"/"salaries.csv")

# merged = pd.merge(employees,salaries,on="EmployeeID")
# print(merged)
# merged = pd.merge( it work for only when two columns have same values
#     employees,
#     salaries,
#     on=["EmployeeID", "Department"]
# )
# print(merged)

df["Bonus"] = df["Salary"] * 0.10
df["Total_Compensation"] = df["Salary"] + df["Bonus"]

# Rename column
df.rename(columns={"EmployeeID":"Emp_ID"}, inplace=True)

# Category
df["Salary_Category"] = df["Salary"].apply(
    lambda salary:
    "High" if salary >= 80000 else "Normal"
)

print(df)

print("\nDepartment Counts")
print(df["Department"].value_counts())

print("\nUnique Departments")
print(df["Department"].unique())

encoded = pd.get_dummies(
    df,
    columns=["Department"]
)

print("\nEncoded Data")
print(encoded.head())