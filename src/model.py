import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,classification_report

# Load cleaned data
data = pd.read_csv("dataset/cleaned_ai4i2020.csv")

X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

best_model = None
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test,pred)

    print("="*60)
    print(name)
    print("Accuracy:",acc)

    print(classification_report(y_test,pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("="*60)
print("Best Accuracy:",best_accuracy)

joblib.dump(best_model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Model Saved Successfully")