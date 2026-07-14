from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder,PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold,RFE,SelectKBest,f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from scipy.stats import boxcox
import numpy as np
import pandas as pd

# standarized and normalized scaling
data = {
    "Age": [20, 25, 30, 35, 40],
    "Salary": [30000, 45000, 60000,70000,90000]
}
df = pd.DataFrame(data=data)
# scaler = StandardScaler()
# df["Standarized_Age"] = scaler.fit_transform(df[["Age"]])
scaler = MinMaxScaler()
# df["Normalized_Age"] = scaler.fit_transform(df[["Age"]])
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,columns=df.columns)
# print(scaled_df)

# Encoding
#Label Encoding
df = pd.DataFrame({
    "Department": ["IT", "HR", "Finance", "IT", "HR"]
})
encoder = LabelEncoder()
df["Encoded"] = encoder.fit_transform(df["Department"])
# print(df)
#Hot One Encoding
encoder = pd.get_dummies(df,columns=["Department"],dtype=int)
# print(encoder)

# Orinal Encoding
education_order = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3
}
df = pd.DataFrame({
    "Education": [
        "Bachelor",
        "Master",
        "PhD",
        "High School"
    ]
})
df["Encoded"] = df["Education"].map(education_order)
# print(df)

# Frequency Encoding

df = pd.DataFrame({
    "City": [
        "Delhi",
        "Delhi",
        "Mumbai",
        "Pune",
        "Delhi",
        "Mumbai"
    ]
})
frequency = df["City"].value_counts()
df["frequency"] = df["City"].map(frequency)
# print(df)

# Target Encoding is replace value with average of target value

# Tranformation
data = np.array([2, 5, 10, 20, 50])

transform,lam = boxcox(data)
# print(transform)
# print(lam)

# Binning

data = {
    "Employee": [
        "Harshit",
        "Rahul",
        "Amit",
        "Priya",
        "Neha",
        "Rohit",
        "Anjali",
        "Vikas",
        "Sneha",
        "Karan"
    ],
    "Age": [18, 22, 27, 31, 35, 42, 48, 55, 60, 65]
}

df = pd.DataFrame(data)
# -----------------------------------
# Step 2: Equal Width Binning
# -----------------------------------

df["Equal_Width_Bin"] = pd.cut(
    df["Age"],
    bins=3
)

# -----------------------------------
# Step 3: Equal Frequency Binning
# -----------------------------------

df["Equal_Frequency_Bin"] = pd.qcut(
    df["Age"],
    q=3
)

# -----------------------------------
# Step 4: Custom Binning
# -----------------------------------

bins = [0, 18, 35, 100]

labels = [
    "Teen",
    "Adult",
    "Senior"
]

df["Custom_Age_Group"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels
)

# Polynomial Features
df = pd.DataFrame({
    "Area": [2, 3],
    "Bedrooms": [1, 2]
})

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

new_data = poly.fit_transform(df)

columns = poly.get_feature_names_out(df.columns)

result = pd.DataFrame(
    new_data,
    columns=columns
)

# print(result)

# Feature Selection
# Filter Selection - Correlation
data = {
    "Age": [25,30,35,40,45],
    "Experience": [1,3,6,10,15],
    "Salary": [30000,40000,55000,70000,90000]
}

df = pd.DataFrame(data)

# print(df.corr())

# Filter Selection - Variance Threshold

df = pd.DataFrame({
    "A":[1,1,1,1],
    "B":[1,2,3,4],
    "C":[10,20,30,40]
})

selector = VarianceThreshold()

selected = selector.fit_transform(df)

# print(selected)

# Wrapper Method - RFE

X = pd.DataFrame({
    "Age":[25,30,35,40,45],
    "Experience":[1,3,6,10,15],
    "Height":[170,172,175,178,180]
})

y = [30000,40000,55000,70000,90000]

model = LinearRegression()

selector = RFE(model, n_features_to_select=2)

selector.fit(X,y)

# print(selector.support_)

# Embedded Method
X = pd.DataFrame({
    "Age":[25,30,35,40,45],
    "Experience":[1,3,6,10,15],
    "Height":[170,172,175,178,180]
})

y = [30000,40000,55000,70000,90000]

model = RandomForestRegressor(random_state=42)

model.fit(X,y)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

# print(importance.sort_values(by="Importance", ascending=False))

# SelectKBest
selector = SelectKBest(score_func=f_regression, k=2)

X_new = selector.fit_transform(X,y)

# print(selector.get_support())

# PCA
df = pd.DataFrame({
    "Height": [160,165,170,175,180],
    "Weight": [55,60,65,70,75]
})

pca = PCA(n_components=1)   
new_data = pca.fit_transform(df)
print(pca.explained_variance_ratio_)
# print(new_data)