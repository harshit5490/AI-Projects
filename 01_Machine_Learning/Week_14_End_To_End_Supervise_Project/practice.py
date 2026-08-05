import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# Load dataset
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR/"data"/"raw"/"WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic inspection
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.describe(include="object"))
# print(df.columns)
# print(df["Churn"].value_counts())

# Standard missing values
# print(df.isnull().sum())
# print((df == "").sum())
# print((df["TotalCharges"].str.strip() == "").sum())

# print(df.duplicated().sum())

# print(df["Churn"].value_counts())
# print(df["Churn"].value_counts(normalize=True) * 100)

# df["Churn"].value_counts().plot(
#     kind="bar",
#     figsize=(6,4)
# )

# plt.title("Customer Churn Distribution")
# plt.xlabel("Churn")
# plt.ylabel("Count")
# plt.show()

# numerical_columns = [
#     "tenure",
#     "MonthlyCharges"
# ]
# df[numerical_columns].describe()
# df[numerical_columns].hist(
#     figsize=(10,4)
# )

# plt.show()

# for col in numerical_columns:

#     plt.figure(figsize=(6,3))

#     plt.boxplot(df[col])

#     plt.title(col)

#     plt.show()

categorical_columns = df.select_dtypes(
    include=["object","string"]
).columns

# for col in categorical_columns:

#     print("="*50)

#     print(col)

#     print(df[col].value_counts())

pd.crosstab(
    df["Contract"],
    df["Churn"]
)    
pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

corr = df[
    [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
].corr()

# print(corr)

# Count
print(pd.crosstab(df["Contract"], df["Churn"]))

# Percentage
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn)

contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Contract Type vs Churn")
plt.xlabel("Contract")
plt.ylabel("Percentage")
plt.legend(title="Churn")
plt.show()

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print(internet_churn)

internet_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Internet Service vs Churn")
plt.ylabel("Percentage")
plt.show()

security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print(security_churn)

security_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Online Security vs Churn")
plt.ylabel("Percentage")
plt.show()

tech_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print(tech_churn)

tech_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Tech Support vs Churn")
plt.ylabel("Percentage")
plt.show()

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print(payment_churn)

payment_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Payment Method vs Churn")
plt.ylabel("Percentage")
plt.show()

paperless_churn = pd.crosstab(
    df["PaperlessBilling"],
    df["Churn"],
    normalize="index"
) * 100

print(paperless_churn)

paperless_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Paperless Billing vs Churn")
plt.ylabel("Percentage")
plt.show()

print(
    df.groupby("Churn")["tenure"].describe()
)

plt.figure(figsize=(6,4))

df.boxplot(
    column="tenure",
    by="Churn"
)

plt.title("Tenure vs Churn")
plt.suptitle("")
plt.show()

print(
    df.groupby("Churn")["MonthlyCharges"].describe()
)

plt.figure(figsize=(6,4))

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title("Monthly Charges vs Churn")
plt.suptitle("")
plt.show()