import os
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import mlflow
import mlflow.sklearn

class ModelTraining:
    def __init__(self, evaluator=None):
        from .evaluate import Evaluator
        self.evaluator = evaluator or Evaluator()
        os.makedirs('models', exist_ok=True)
        mlflow.set_experiment('CustomerChurnPrediction')

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            C = trial.suggest_loguniform('C', 1e-3, 10)
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
            model = LogisticRegression(C=C, solver=solver, max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return 1.0 - f1_score(y_test, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        model = LogisticRegression(**best_params, max_iter=1000)
        with mlflow.start_run(run_name='LogisticRegression'):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = self.evaluator.evaluate_model(y_test, preds, proba)
            mlflow.log_params(best_params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'confusion_matrix'})
            mlflow.sklearn.log_model(model, 'model')
        with open('models/logistic_regression.pkl', 'wb') as f:
            pickle.dump(model, f)
        return metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 2, 16)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return 1.0 - f1_score(y_test, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        model = RandomForestClassifier(**best_params, random_state=42)
        with mlflow.start_run(run_name='RandomForest'):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = self.evaluator.evaluate_model(y_test, preds, proba)
            mlflow.log_params(best_params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'confusion_matrix'})
            mlflow.sklearn.log_model(model, 'model')
        with open('models/random_forest.pkl', 'wb') as f:
            pickle.dump(model, f)
        return metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'random_state': 42
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return 1.0 - f1_score(y_test, preds)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        best_params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'use_label_encoder': False, 'random_state': 42})
        model = xgb.XGBClassifier(**best_params)
        with mlflow.start_run(run_name='XGBoost'):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics = self.evaluator.evaluate_model(y_test, preds, proba)
            mlflow.log_params(best_params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'confusion_matrix'})
            mlflow.sklearn.log_model(model, 'model')
        with open('models/xgboost.pkl', 'wb') as f:
            pickle.dump(model, f)
        return metrics

    def cross_validate_model(self, train_func, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics_list = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            metrics = train_func(X_train, y_train, X_test, y_test)
            metrics_list.append(metrics)
        avg_metrics = {}
        for key in metrics_list[0]:
            if key == 'confusion_matrix':
                avg_metrics[key] = np.mean([np.array(m[key]) for m in metrics_list], axis=0).tolist()
            else:
                avg_metrics[key] = float(np.mean([m[key] for m in metrics_list if m[key] is not None]))
        return avg_metrics
