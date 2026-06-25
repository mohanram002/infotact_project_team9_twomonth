import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data= pd.read_csv("dataset/ai4i2020.csv")

print(data["Machine failure"].value_counts())
print(data["Machine failure"].value_counts(normalize=True)*100)

print(data.isnull().sum())

data = data.drop(columns=["UDI", "Product ID"])

print(data["Type"].unique())


le = LabelEncoder()
data["Type"] = le.fit_transform(data["Type"])



plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()