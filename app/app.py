import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from src.feature_engineering import FeatureEngineering
from src.data_preprocessing import DataPreprocessing
from src.model_training import ModelTraining
import shap

app = Flask(__name__, template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates')))

METRICS_PATH = os.path.join('metrics', 'metrics.pkl')
MODEL_NAMES = ['logistic_regression', 'random_forest', 'xgboost']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    raw_path = os.path.join('data', 'raw', 'raw.csv')
    df = pd.read_csv(raw_path)
    fe = FeatureEngineering()
    df = fe.handle_missing_values(df)
    df = fe.add_valuable_features(df)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_path = os.path.join('data', 'processed', 'train.csv')
    test_path = os.path.join('data', 'processed', 'test.csv')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    dp = DataPreprocessing()
    X_train, X_test = dp.preprocess_and_save(train_path, test_path)
    y_train = X_train[:, -1]
    y_test = X_test[:, -1]
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]
    mt = ModelTraining()
    metrics = {}
    metrics['logistic_regression'] = mt.train_logistic_regression(X_train, y_train, X_test, y_test)
    metrics['random_forest'] = mt.train_random_forest(X_train, y_train, X_test, y_test)
    metrics['xgboost'] = mt.train_xgboost(X_train, y_train, X_test, y_test)
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
    fe = FeatureEngineering()
    df = pd.DataFrame([input_data])
    df = fe.handle_missing_values(df)
    df = fe.add_valuable_features(df)
    df = df.drop(columns=['customerID'], errors='ignore')
    dp = DataPreprocessing()
    with open(dp.config.preprocessor_obj_file_path, 'rb') as f:
        preprocessor = pickle.load(f)
    X_processed = preprocessor.transform(df)
    model_path = os.path.join('models', f'{model_name}.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    proba = model.predict_proba(X_processed)[0, 1]
    pred = model.predict(X_processed)[0]
    churn_label = 'Yes' if pred == 1 else 'No'
    if model_name == 'xgboost' or model_name == 'random_forest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        if isinstance(shap_values, list):
            shap_row = shap_values[1][0]
        else:
            if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
                shap_row = shap_values[0, :, 1]
            else:
                shap_row = shap_values[0]
        feature_names = preprocessor.get_feature_names_out()
        shap_dict = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:5]
    else:
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