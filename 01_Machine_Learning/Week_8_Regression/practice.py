import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import PolynomialFeatures

# Linear Regression
# data = {
#     "Hours": [1,2,3,4,5,6,7,8,9,10],
#     "Marks": [40,45,50,60,65,70,75,85,90,95]
# }

# df = pd.DataFrame(data)

# X = df[["Hours"]]
# y = df["Marks"]

# X_train,X_test,y_train,y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )
# model = LinearRegression()
# model.fit(X_train,y_train)
# print(model.coef_[0]) #slope
# print(model.intercept_) #intercept

# # prediction = model.predict([[8]])
# # print(prediction)

# y_pred = model.predict(X_test)
# result = pd.DataFrame({
#     "Actual": y_test,
#     "Predicted": y_pred
# })

# print(result)

# plt.scatter(X, y, label="Actual Data")

# plt.plot(X, model.predict(X), color="red", label="Regression Line")

# plt.xlabel("Hours Studied")
# plt.ylabel("Marks")
# plt.title("Linear Regression")

# plt.legend()
# plt.grid(True)

# plt.show()

# mae = mean_absolute_error(y_test, y_pred)

# mse = mean_squared_error(y_test, y_pred)

# rmse = np.sqrt(mse)

# r2 = r2_score(y_test, y_pred)

# print("MAE :", mae)
# print("MSE :", mse)
# print("RMSE:", rmse)
# print("R²  :", r2)

# Multiple Linear Regression

# data = {
#     "Area":[1000,1200,1500,1800,2000],
#     "Bedrooms":[2,3,3,4,4],
#     "Age":[10,8,6,5,2],
#     "Price":[250000,300000,350000,420000,480000]
# }

# df = pd.DataFrame(data)

# X = df[["Area","Bedrooms","Age"]]
# y = df["Price"]

# X_train,X_test,y_train,y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )
# model = LinearRegression()
# model.fit(X_train,y_train)
# print(model.coef_)
# print(model.intercept_)
# prediction=model.predict([[1600,3,4]])

# # print(prediction)

# y_pred=model.predict(X_test)

# print(y_pred)

# print("MAE :",mean_absolute_error(y_test,y_pred))

# mse=mean_squared_error(y_test,y_pred)

# print("MSE :",mse)

# print("RMSE :",np.sqrt(mse))

# print("R2 :",r2_score(y_test,y_pred))

# Polynomial Regression

# data = {
#     "Experience":[1,2,3,4,5,6,7,8],
#     "Salary":[30000,35000,45000,60000,82000,110000,145000,190000]
# }

# df = pd.DataFrame(data)

# X = df[["Experience"]]
# y = df["Salary"]
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)
# model = LinearRegression()

# model.fit(X_poly,y)

# prediction = model.predict(poly.transform([[5]]))

# print(prediction)

# plt.scatter(X,y)

# plt.plot(
#     X,
#     model.predict(X_poly),
#     color="red"
# )

# plt.title("Polynomial Regression")

# plt.xlabel("Experience")

# plt.ylabel("Salary")

# plt.show()
