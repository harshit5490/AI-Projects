import pandas as pd
import numpy as np

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
print(df.describe())