from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    values = [
    float(request.form["type"]),
    float(request.form["air_temp"]),
    float(request.form["process_temp"]),
    float(request.form["rpm"]),
    float(request.form["torque"]),
    float(request.form["tool_wear"]),
    float(request.form["twf"]),
    float(request.form["hdf"]),
    float(request.form["pwf"]),
    float(request.form["osf"]),
    float(request.form["rnf"])
    
]

    data = scaler.transform([values])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Machine Failure"
    else:
        result = "Healthy Machine"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)