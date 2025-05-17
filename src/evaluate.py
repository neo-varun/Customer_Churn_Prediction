import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class Evaluator:
    def evaluate_model(self, y_true, y_pred, y_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['roc_auc'] = None
        return metrics
