# Customer Churn Prediction App

## Overview
This project is a Customer Churn Prediction Application built using Flask, scikit-learn, XGBoost, Optuna, and SHAP. It predicts customer churn based on tabular customer data, providing model training, evaluation, and interpretability through a modern web interface. The application supports multiple models and offers detailed analytics and SHAP-based explanations for predictions.

## Features
- **Churn Prediction** – Predict whether a customer will churn based on their profile
- **Multiple Model Support** – Train and compare Logistic Regression, Random Forest, and XGBoost models
- **Interactive Web Interface** – User-friendly UI for training, prediction, and metrics visualization
- **Model Training Interface** – Train models directly from the web interface
- **Comprehensive Analytics** – View detailed model performance metrics (accuracy, precision, recall, F1, ROC AUC, confusion matrix)
- **Explainable AI** – SHAP explanations for model predictions

## Prerequisites

### Data Preparation
Before running the application, you need a customer churn dataset (CSV format) with features similar to the Telco Customer Churn dataset. Place your raw data as:
```
data/
└── raw/
    └── raw.csv
```

### System Requirements
- Python 3.8+

## Installation & Setup

### Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
1. Activate your virtual environment
2. Launch the Flask server:
```bash
python app/app.py
```
3. Open your web browser and navigate to: `http://localhost:8000`

## How the Program Works

### Application Components
1. **Feature Engineering (`src/feature_engineering.py`)**
   - Handles missing values and creates new features for better model performance
2. **Data Preprocessing (`src/data_preprocessing.py`)**
   - Scales numerical features and encodes categorical features
   - Saves and loads preprocessing pipelines
3. **Model Training (`src/model_training.py`)**
   - Supports Logistic Regression, Random Forest, and XGBoost
   - Hyperparameter tuning with Optuna
   - Model evaluation and saving
4. **Evaluation (`src/evaluate.py`)**
   - Calculates accuracy, precision, recall, F1, ROC AUC, and confusion matrix
5. **Web Interface (`templates/index.html`, `app/app.py`)**
   - Train models, view metrics, and make predictions interactively
   - SHAP explanations for prediction interpretability

### Performance Metrics & Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

### Usage Guide

1. **Data Preparation**
   - Place your raw customer data as `data/raw/raw.csv`
   - Ensure the data columns match the expected features
2. **Model Training**
   - Click "Train Model" on the web interface
   - Wait for training to complete and view metrics
3. **Prediction**
   - Enter customer details in the form
   - Select a model and click "Predict"
   - View churn prediction, probability, and top SHAP features

## Technologies Used
- **Flask** (Web Server)
- **scikit-learn** (ML Models & Preprocessing)
- **XGBoost** (Gradient Boosted Trees)
- **Optuna** (Hyperparameter Optimization)
- **SHAP** (Model Explainability)
- **pandas, numpy** (Data Processing)

## Performance Metrics
The app provides:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

## License
This project is licensed under the MIT License.

## Author
Developed by Varun. Feel free to connect with me:
- Email: darklususnaturae@gmail.com
