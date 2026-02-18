# 🏠 House Price Prediction Project

## 📌 Overview
This project aims to predict house prices based on features such as location, size, number of rooms, condition, and other property-related characteristics using Machine Learning regression models.

The project demonstrates **data preprocessing, feature engineering, model training, evaluation, and model comparison**.

---

## 🎯 Objective
Build a regression model to accurately estimate house prices and identify the key factors influencing property value.

---

## 📊 Dataset Features
- CRIM: Crime rate by town
- ZN: Residential land zoned for lots over 25,000 sq.ft.
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: Nitric oxides concentration
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to radial highways
- TAX: Full-value property-tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk − 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population
- MEDV: Median value of owner-occupied homes in $1000s (Target Variable)

---

## 🛠️ Project Workflow

1. **Data Cleaning**
   - Removed duplicates
   - Handled missing values

2. **Data Preprocessing**
   - Feature scaling
   - One-Hot Encoding for categorical variables (if any)
   - Remove outliers
   - Feature engineering (e.g., TotalSF)

3. **Model Training**
   - Single Feature Linear Regression
   - Multiple Linear Regression
   - Optional Ridge Regression

4. **Model Evaluation**
   - RMSE (Root Mean Squared Error)
   - R² Score

---

## 📈 Results
Linear Regression provided a strong baseline model.  
However, due to non-linear relationships in housing features, tree-based models such as **Decision Tree** and **Random Forest** are expected to perform better, achieving higher R² and lower RMSE.

---

## 🚀 Future Improvements
- Implement Decision Tree & Random Forest models
- Hyperparameter tuning
- Feature importance analysis
- Deploy the model using Streamlit or Flask

---

## 🧠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Joblib

---

## 📂 Project Structure
