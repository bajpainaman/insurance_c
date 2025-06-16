"""
Data Processing Module for Fraud Detection
Handles data loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from .constants import *


class DataProcessor:
    """Handles all data processing operations"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_columns = None
        self.is_fitted = False
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            df['fraud_label'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for fraud detection"""
        enhanced_df = df.copy()
        
        # Ratio features
        if 'total_claim_amount' in enhanced_df.columns and 'policy_annual_premium' in enhanced_df.columns:
            enhanced_df['claim_to_premium_ratio'] = (
                enhanced_df['total_claim_amount'] / (enhanced_df['policy_annual_premium'] + 1e-6)
            )
            enhanced_df['premium_efficiency'] = (
                enhanced_df['policy_annual_premium'] / (enhanced_df['total_claim_amount'] + 1e-6)
            )
        
        # Time-based features
        if 'months_as_customer' in enhanced_df.columns:
            enhanced_df['customer_tenure_log'] = np.log1p(enhanced_df['months_as_customer'])
            enhanced_df['is_new_customer'] = (enhanced_df['months_as_customer'] < 12).astype(int)
            enhanced_df['is_long_term_customer'] = (enhanced_df['months_as_customer'] > 60).astype(int)
        
        # Age-based features
        if 'age' in enhanced_df.columns:
            enhanced_df['age_squared'] = enhanced_df['age'] ** 2
            enhanced_df['age_risk_score'] = np.where(
                (enhanced_df['age'] < 25) | (enhanced_df['age'] > 70), 1, 0
            )
            enhanced_df['age_normalized'] = (
                (enhanced_df['age'] - enhanced_df['age'].mean()) / enhanced_df['age'].std()
            )
            enhanced_df['age_bin'] = pd.cut(
                enhanced_df['age'], bins=AGE_BINS, labels=AGE_LABELS
            )
        
        # Interaction features
        if 'months_as_customer' in enhanced_df.columns and 'total_claim_amount' in enhanced_df.columns:
            enhanced_df['customer_loyalty_claim'] = (
                enhanced_df['months_as_customer'] * enhanced_df['total_claim_amount']
            )
        
        # Geographic risk features
        if 'incident_state' in enhanced_df.columns:
            state_fraud_rates = enhanced_df.groupby('incident_state')['fraud_label'].mean()
            enhanced_df['state_fraud_risk'] = enhanced_df['incident_state'].map(state_fraud_rates)
        
        # Vehicle-related features
        if 'vehicle_claim' in enhanced_df.columns and 'property_claim' in enhanced_df.columns:
            enhanced_df['total_claim_amount_enhanced'] = (
                enhanced_df['vehicle_claim'] + enhanced_df['property_claim']
            )
            enhanced_df['claim_complexity'] = (
                (enhanced_df['vehicle_claim'] > 0).astype(int) + 
                (enhanced_df['property_claim'] > 0).astype(int)
            )
        
        # Binning continuous variables
        self._create_quantile_features(enhanced_df)
        
        return enhanced_df
    
    def _create_quantile_features(self, df: pd.DataFrame) -> None:
        """Create quantile-based features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            if col not in ['fraud_label', 'fraud_reported']:
                try:
                    df[f'{col}_binned'] = pd.qcut(
                        df[col], q=QUANTILE_BINS, labels=QUANTILE_LABELS, duplicates='drop'
                    )
                except ValueError:
                    df[f'{col}_binned'] = pd.cut(
                        df[col], bins=QUANTILE_BINS, labels=QUANTILE_LABELS
                    )
    
    def prepare_preprocessing_pipeline(self, df: pd.DataFrame) -> None:
        """Prepare preprocessing pipeline"""
        self.feature_columns = df.columns.drop(['fraud_reported', 'fraud_label']).tolist()
        
        numeric_features = df[self.feature_columns].select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        categorical_features = df[self.feature_columns].select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        
        # Preprocessing pipelines
        string_cast_transformer = FunctionTransformer(
            lambda X: X.astype(str), validate=False
        )
        
        numeric_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('cast', string_cast_transformer),
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ], remainder='drop')
    
    def split_and_preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Split data and apply preprocessing"""
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=TEST_SIZE, 
            stratify=df['fraud_label'], 
            random_state=RANDOM_SEED
        )
        
        # Fit preprocessing pipeline
        X_train = self.preprocessor.fit_transform(train_df[self.feature_columns])
        X_val = self.preprocessor.transform(val_df[self.feature_columns])
        
        y_train = train_df['fraud_label'].values
        y_val = val_df['fraud_label'].values
        
        self.is_fitted = True
        
        return X_train, X_val, y_train, y_val, train_df, val_df
    
    def apply_smote_balancing(self, X_train, y_train) -> tuple:
        """Apply SMOTE for class balancing"""
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=SMOTE_K_NEIGHBORS)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Convert to dense if sparse
        if hasattr(X_train_balanced, 'toarray'):
            X_train_balanced = X_train_balanced.toarray()
        
        return X_train_balanced, y_train_balanced
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise Exception("Preprocessor not fitted. Call split_and_preprocess_data first.")
        
        enhanced_df = self.create_enhanced_features(df)
        X_transformed = self.preprocessor.transform(enhanced_df[self.feature_columns])
        
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        return X_transformed