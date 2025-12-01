# 🩺 Diabetes Prediction – Streamlit Demo

This project is a simple simulation of **machine learning model deployment** using **Streamlit** and a trained model saved with **joblib**.  
The goal of this application is to predict the likelihood of diabetes based on several clinical measurement features.

---

## 📊 Dataset Description

The dataset used in this project originates from the **National Institute of Diabetes and Digestive and Kidney Diseases**, commonly referred to as the *Pima Indians Diabetes Dataset*.

The primary objective of this dataset is to **diagnostically predict whether a patient has diabetes**, based on several medical predictor variables (independent features) and one target variable:

| Target Variable | Meaning |
|-----------------|----------|
| Outcome = 0 | Non-diabetic |
| Outcome = 1 | Diabetic |

The dataset includes the following features:

- Gender  
- Age  
- Hypertension  
- Heart disease  
- Smoking history  
- BMI (Body Mass Index)  
- HbA1c level  
- Blood glucose level  

> Kaggle dataset source for licensing information and extended dataset description.
[here](https://www.kaggle.com/code/mehmetisik/diabetes-eda-ml-prediction/notebook)

---

## 🧠 Project Overview

This project consists of:

1. Training a machine learning model using the diabetes dataset
2. Saving the trained model as `model.joblib`
3. Deploying the model in a web application built with **Streamlit**

The application enables users to input patient information manually and receive a prediction indicating whether the patient is likely to have diabetes.

---

## ⚙ Installation
1️⃣ Clone the repository or create the project directory
git clone <YOUR_REPOSITORY_URL>
cd <project-folder>

2️⃣ Create and activate a virtual environment (recommended)
python -m venv venv
# activate
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

3️⃣ Install the dependencies

## Install manually:
```
pip install streamlit pandas scikit-learn joblib numpy
```
If training dependencies are needed:
```
pip install imbalanced-learn matplotlib seaborn jupyter
```
Or install from requirements file (if available):
```
pip install -r requirements.txt
```

## ▶ Running the Streamlit Application
Make sure app.py and model.joblib are located in the same directory.

Start the application using:
```
streamlit run app.py
```
After running the command, Streamlit will provide a local URL such as:
```
http://localhost:8501
```
Open this URL in your browser to interact with the prediction interface.
