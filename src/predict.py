import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

sample = {
    "Type":0,
    "Air temperature [K]":298.1,
    "Process temperature [K]":308.6,
    "Rotational speed [rpm]":1551,
    "Torque [Nm]":42.8,
    "Tool wear [min]":0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0
    
}

df = pd.DataFrame([sample])

scaled = scaler.transform(df)

prediction = model.predict(scaled)

if prediction[0] == 1:
    print("Machine Failure")
else:
    print("Machine is Healthy")