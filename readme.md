# Predictive Maintenance for Industrial Equipment

📌 Project Overview

This project is developed as part of the **Infotact Internship Program**. The objective is to predict whether an industrial machine is likely to fail based on its operating conditions using Machine Learning.

The project performs data preprocessing, exploratory data analysis (EDA), model training, and provides a prediction interface using a Flask web application.

---

## 🚀 Features

* Exploratory Data Analysis (EDA)
* Data Cleaning & Preprocessing
* Feature Scaling using StandardScaler
* Machine Learning Model Training
* Model Evaluation
* Predictive Maintenance Web Application
* Real-time Machine Failure Prediction

---

## 🛠️ Technologies Used

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Flask
* Joblib

---


## 📊 Dataset

Dataset: **AI4I 2020 Predictive Maintenance Dataset**

The dataset contains machine operating parameters such as:

* Air Temperature
* Process Temperature
* Rotational Speed (RPM)
* Torque
* Tool Wear
* Machine Type

Target Variable:

* Machine Failure (0 = No Failure, 1 = Failure)

---

## ⚙️ Project Workflow

### 1. Data Collection

Load the AI4I 2020 Predictive Maintenance dataset.

### 2. Exploratory Data Analysis

* Check missing values
* Remove unnecessary columns
* Visualize feature distributions
* Analyze feature correlations
* Detect outliers

### 3. Data Preprocessing

* Encode categorical variables
* Feature Scaling using StandardScaler
* Train-Test Split

### 4. Model Training

Train a **logistic regression** using the processed dataset.

### 5. Model Evaluation

Evaluate the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### 6. Model Saving

Save the trained model and scaler using Joblib.

### 7. Prediction

Load the saved model and scaler to predict machine failure for new input data.



## 📈 Machine Learning Model

**Model Used:**

* Logistic regression

Reason for choosing Logistic regression:

* High accuracy
* Handles non-linear relationships
* Less prone to overfitting
* Works well with tabular industrial datasets

---

Then include them like:

```markdown
![Home Page](screenshots/home_page.png)

![Correlation Heatmap](screenshots/eda_heatmap.png)

![Prediction Result](screenshots/prediction_result.png)
```

---

## 👨‍💻 Team Members

* Mohan Ram G
* Hariharan M

---

## 📌 Future Improvements

* Deploy the application online
* Add multiple ML model comparison
* Improve UI/UX
* Add prediction history
* Integrate database support

---

## 📄 License

This project is developed for educational and internship purposes.

---

## ⭐ Acknowledgement

Developed during the **Infotact Internship Program** using the AI4I 2020 Predictive Maintenance dataset.
