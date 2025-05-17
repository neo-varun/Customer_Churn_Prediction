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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)