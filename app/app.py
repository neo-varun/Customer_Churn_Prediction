import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
import os
import pickle
import pandas as pd
from src.feature_engineering import FeatureEngineering
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
import shap

app = Flask(__name__, template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates')))

# Store metrics in memory for demo (could be saved to disk)
METRICS_PATH = os.path.join('metrics', 'metrics.pkl')
MODEL_NAMES = ['logistic_regression', 'random_forest', 'xgboost']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    # Feature engineering
    raw_path = os.path.join('data', 'raw', 'raw.csv')
    df = pd.read_csv(raw_path)
    fe = FeatureEngineering()
    df = fe.handle_missing_values(df)
    df = fe.add_valuable_features(df)
    # Split train/test (simple split for demo)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_path = os.path.join('data', 'processed', 'train.csv')
    test_path = os.path.join('data', 'processed', 'test.csv')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Preprocessing
    dp = DataPreprocessing()
    X_train, X_test = dp.preprocess_and_save(train_path, test_path)
    y_train = X_train[:, -1]
    y_test = X_test[:, -1]
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]
    # Train models
    mt = ModelTraining()
    metrics = {}
    metrics['logistic_regression'] = mt.train_logistic_regression(X_train, y_train, X_test, y_test)
    metrics['random_forest'] = mt.train_random_forest(X_train, y_train, X_test, y_test)
    metrics['xgboost'] = mt.train_xgboost(X_train, y_train, X_test, y_test)
    # Save metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)
    return jsonify({'message': 'Training complete!'}), 200

@app.route('/metrics')
def metrics():
    model = request.args.get('model')
    if not os.path.exists(METRICS_PATH):
        return jsonify({'models': [], 'metrics': {}})
    with open(METRICS_PATH, 'rb') as f:
        all_metrics = pickle.load(f)
    if not model:
        return jsonify({'models': list(all_metrics.keys()), 'metrics': {}})
    if model not in all_metrics:
        return jsonify({'metrics': {}})
    return jsonify({'metrics': all_metrics[model]})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model')
    input_data = data.get('input')
    # Prepare DataFrame for a single row
    columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
        'tenure_group', 'total_services', 'is_senior', 'has_family', 'avg_monthly_charge',
        'is_long_term_contract', 'has_streaming', 'has_support_or_security', 'payment_type'
    ]
    # If frontend sends only raw.csv fields, do feature engineering
    fe = FeatureEngineering()
    df = pd.DataFrame([input_data])
    df = fe.handle_missing_values(df)
    df = fe.add_valuable_features(df)
    # Drop customerID if present
    df = df.drop(columns=['customerID'], errors='ignore')
    # Preprocessing
    dp = DataPreprocessing()
    with open(dp.config.preprocessor_obj_file_path, 'rb') as f:
        preprocessor = pickle.load(f)
    X_processed = preprocessor.transform(df)
    # Load model
    model_path = os.path.join('models', f'{model_name}.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    proba = model.predict_proba(X_processed)[0, 1]
    pred = model.predict(X_processed)[0]
    churn_label = 'Yes' if pred == 1 else 'No'
    # SHAP explanation
    if model_name == 'xgboost' or model_name == 'random_forest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        # Debug: print SHAP values shape and sample
        print('SHAP values type:', type(shap_values))
        if isinstance(shap_values, list):
            print('SHAP values[1] shape:', getattr(shap_values[1], 'shape', None))
            print('SHAP values[1][0] sample:', shap_values[1][0][:5])
            shap_row = shap_values[1][0]  # class 1, first row
        else:
            print('SHAP values shape:', getattr(shap_values, 'shape', None))
            print('SHAP values[0] sample:', shap_values[0][:5])
            # For shape (1, n_features, 2): [sample, feature, class]
            if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
                shap_row = shap_values[0, :, 1]  # class 1, first row
            else:
                shap_row = shap_values[0]  # fallback
        feature_names = preprocessor.get_feature_names_out()
        shap_dict = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:5]
    else:  # logistic_regression
        # Use training data as background for SHAP
        train_path = os.path.join('data', 'processed', 'train.csv')
        train_df = pd.read_csv(train_path)
        train_df = fe.handle_missing_values(train_df)
        train_df = fe.add_valuable_features(train_df)
        train_df = train_df.drop(columns=['customerID'], errors='ignore')
        X_train_bg = preprocessor.transform(train_df)
        explainer = shap.LinearExplainer(model, X_train_bg, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_processed)
        shap_row = shap_values[0] if len(shap_values.shape) == 2 else shap_values
        feature_names = preprocessor.get_feature_names_out()
        shap_dict = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:5]
    shap_explanation = [{'feature': k, 'value': float(v)} for k, v in shap_dict]
    return jsonify({
        'churn': churn_label,
        'probability': float(proba),
        'shap': shap_explanation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)