# PROJECTTT

# Diabetes Risk Prediction System 

## What it does

This is our course project for predicting diabetes risk from age, gender, BMI, HbA1c, blood glucose, hypertension, heart disease, and smoking history. We used Kaggle’s Diabetes Prediction Dataset by iammustafatz. It has 100k rows and around 8.5% diabetes cases.

We removed duplicates and rare `Other` gender rows, encoded categorical values, scaled numeric columns, and used an 80:20 stratified split. In Short cleaned the whole dataset. and performed EDA then started training.

## Results

We trained Logistic Regression, KNN, Naive Bayes, Decision Tree, and Random Forest. KNN did best among the basic models with recall `0.6203`, F1 `0.7159`, and ROC-AUC `0.9010`. Logistic Regression got `0.8857` accuracy but was not great for this data. More experiments like GB, XGB, and Voting Classifier are in the `experimenting` branch.

This is a course project, not a medical tool.

## Run it

```bash
pip install -r requirements.txt
```

Open `notebook.ipynb` and run all cells.

```text
notebook.ipynb
preprocessing.py
models/
requirements.txt
```