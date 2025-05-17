import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join('pipeline', 'churn_preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.config = DataPreprocessingConfig()
        self.numerical_features = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'total_services', 'is_senior', 'has_family', 'avg_monthly_charge',
            'is_long_term_contract', 'has_streaming', 'has_support_or_security'
        ]
        self.categorical_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'tenure_group', 'payment_type'
        ]
        self.columns_to_drop = ['customerID']

    def get_preprocessor(self):
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.numerical_features),
            ('cat', cat_pipeline, self.categorical_features)
        ])
        return preprocessor

    def preprocess_and_save(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Drop unnecessary columns
        for col in self.columns_to_drop:
            train_df.drop(columns=[col], inplace=True, errors='ignore')
            test_df.drop(columns=[col], inplace=True, errors='ignore')

        # Log data shape
        logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        target_column = 'Churn'
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column].map({'Yes': 1, 'No': 0})
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column].map({'Yes': 1, 'No': 0})

        preprocessor = self.get_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Save preprocessor
        os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)
        with open(self.config.preprocessor_obj_file_path, 'wb') as f:
            pickle.dump(preprocessor, f)

        # Save processed data (train + test) as processed.csv
        os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
        processed_train = pd.DataFrame(X_train_processed)
        processed_train['Churn'] = y_train.values
        processed_test = pd.DataFrame(X_test_processed)
        processed_test['Churn'] = y_test.values
        processed_df = pd.concat([processed_train, processed_test], ignore_index=True)
        processed_df.to_csv(os.path.join('data', 'processed', 'processed.csv'), index=False)

        train_arr = np.c_[X_train_processed, y_train.values]
        test_arr = np.c_[X_test_processed, y_test.values]
        return train_arr, test_arr

    def load_and_preprocess(self, data_path):
        df = pd.read_csv(data_path)
        for col in self.columns_to_drop:
            df.drop(columns=[col], inplace=True, errors='ignore')
        target = 'Churn'
        X = df.drop(columns=[target])
        y = df[target].map({'Yes': 1, 'No': 0}).values
        with open(self.config.preprocessor_obj_file_path, 'rb') as f:
            preprocessor = pickle.load(f)
        X_processed = preprocessor.transform(X)
        return X_processed, y
