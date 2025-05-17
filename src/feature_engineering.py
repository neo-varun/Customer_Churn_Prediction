import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class FeatureEngineering:
    def __init__(self):
        self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numerical_features = self.numerical_features
        categorical_features = self.categorical_features
        # Convert numerical columns to numeric, coercing errors to NaN
        for col in numerical_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Numerical imputation (median)
        num_imputer = SimpleImputer(strategy='median')
        df[numerical_features] = num_imputer.fit_transform(df[numerical_features])
        # Categorical imputation (most frequent)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])
        return df

    def add_valuable_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Tenure group
        df['tenure_group'] = pd.cut(df['tenure'],
                                    bins=[-1, 12, 24, 48, 60, np.inf],
                                    labels=['0-12', '13-24', '25-48', '49-60', '61+'])
        # Total services count
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        df['total_services'] = df[service_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
        # Is senior citizen (already binary, but ensure type)
        df['is_senior'] = df['SeniorCitizen'].astype(int)
        # Has dependents and partner (interaction)
        df['has_family'] = ((df['Partner'] == 'Yes') & (df['Dependents'] == 'Yes')).astype(int)
        # Average monthly charge (handle division by zero)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['avg_monthly_charge'] = df.apply(lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'], axis=1)
        # Is long-term contract
        df['is_long_term_contract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
        # Has streaming services
        df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
        # Has tech support or online security
        df['has_support_or_security'] = ((df['TechSupport'] == 'Yes') | (df['OnlineSecurity'] == 'Yes')).astype(int)
        # Payment method type (extract main type)
        df['payment_type'] = df['PaymentMethod'].str.extract(r'^(\w+)')[0].str.lower()
        return df
