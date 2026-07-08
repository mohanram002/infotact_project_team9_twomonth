import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv("dataset/cleaned_ai4i2020.csv")

print("="*50)
print("Dataset Shape")
print(data.shape)

print("="*50)
print("First Five Rows")
print(data.head())

print("="*50)
print("Dataset Info")
print(data.info())

print("="*50)
print("Missing Values")
print(data.isnull().sum())

print("="*50)
print("Class Distribution")
print(data["Machine failure"].value_counts())

print(data["Machine failure"].value_counts(normalize=True)*100)

# Drop unnecessary columns
data.drop(columns=["UDI","Product ID"], inplace=True)

# Encode categorical feature
le = LabelEncoder()
data["Type"] = le.fit_transform(data["Type"])

print("="*50)
print("Statistical Summary")
print(data.describe())

# Target Distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Machine failure", data=data)
plt.title("Machine Failure Distribution")
plt.show()

# Histograms
data.hist(figsize=(14,10))
plt.tight_layout()
plt.show()

# Boxplots
for col in data.columns[:-1]:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=data[col])
    plt.title(col)
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

