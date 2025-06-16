"""
Ensemble Methods for Fraud Detection
Implements multiple model variants, stacking, and cross-validation
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from .constants import *


class EnsembleSystem:
    """Comprehensive ensemble system with all 4 components"""
    
    def __init__(self):
        self.base_models = self._create_base_models()
        self.trained_models = {}
        self.meta_learner = None
        self.model_weights = {}
        self.cross_validation_results = {}
    
    def _create_base_models(self) -> dict:
        """Create diverse model variants"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=ENSEMBLE_N_ESTIMATORS,
                max_depth=ENSEMBLE_MAX_DEPTH,
                min_samples_split=5,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=ENSEMBLE_N_ESTIMATORS,
                learning_rate=0.1,
                max_depth=6,
                random_state=RANDOM_SEED
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=RANDOM_SEED,
                max_iter=1000
            ),
            'svm': SVC(
                probability=True,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                kernel='rbf'
            ),
            'lightgbm': lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=ENSEMBLE_N_ESTIMATORS,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                learning_rate=0.05,
                n_estimators=ENSEMBLE_N_ESTIMATORS,
                max_depth=6,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        }
    
    def perform_cross_validation(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Perform stratified cross-validation on all models"""
        print("🔄 Performing cross-validation...")
        
        stratified_kfold = StratifiedKFold(
            n_splits=ENSEMBLE_CV_FOLDS, 
            shuffle=True, 
            random_state=RANDOM_SEED
        )
        
        for model_name, model in self.base_models.items():
            print(f"   Cross-validating {model_name}...")
            
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=stratified_kfold,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                self.cross_validation_results[model_name] = {
                    'mean_auc': cv_scores.mean(),
                    'std_auc': cv_scores.std(),
                    'individual_scores': cv_scores
                }
                
                print(f"     CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"     ❌ Cross-validation failed: {str(e)[:50]}...")
                continue
        
        return self.cross_validation_results
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_validation: np.ndarray, y_validation: np.ndarray) -> dict:
        """Train all base models and evaluate performance"""
        print("🤖 Training base models...")
        
        model_scores = {}
        
        for model_name, model in self.base_models.items():
            print(f"   Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
                
                # Evaluate on validation set
                validation_predictions = model.predict_proba(X_validation)[:, 1]
                validation_auc = roc_auc_score(y_validation, validation_predictions)
                model_scores[model_name] = validation_auc
                
                print(f"     Validation AUC: {validation_auc:.4f}")
                
            except Exception as e:
                print(f"     ❌ Training failed: {str(e)[:50]}...")
                continue
        
        return model_scores
    
    def create_stacking_meta_learner(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_validation: np.ndarray, y_validation: np.ndarray) -> dict:
        """Create stacking meta-learner with out-of-fold predictions"""
        print("🏗️ Creating stacking meta-learner...")
        
        # Generate out-of-fold predictions
        num_models = len(self.trained_models)
        num_samples = len(X_train)
        meta_features_train = np.zeros((num_samples, num_models))
        meta_features_validation = np.zeros((len(X_validation), num_models))
        
        stratified_kfold = StratifiedKFold(
            n_splits=ENSEMBLE_CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED
        )
        
        model_names = list(self.trained_models.keys())
        
        # Generate meta-features using cross-validation
        for fold_idx, (train_indices, validation_indices) in enumerate(stratified_kfold.split(X_train, y_train)):
            print(f"   Processing fold {fold_idx + 1}/{ENSEMBLE_CV_FOLDS}...")
            
            X_fold_train = X_train[train_indices]
            X_fold_validation = X_train[validation_indices]
            y_fold_train = y_train[train_indices]
            
            for model_idx, (model_name, model_class) in enumerate(self.base_models.items()):
                if model_name not in self.trained_models:
                    continue
                
                try:
                    # Create fresh model instance for this fold
                    fold_model = model_class.__class__(**model_class.get_params())
                    
                    # Train on fold training data
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Predict on fold validation data
                    fold_predictions = fold_model.predict_proba(X_fold_validation)[:, 1]
                    meta_features_train[validation_indices, model_idx] = fold_predictions
                    
                except Exception as e:
                    print(f"     ⚠️ {model_name} failed in fold {fold_idx}: {str(e)[:30]}...")
                    meta_features_train[validation_indices, model_idx] = 0.5  # Default prediction
        
        # Generate meta-features for validation set using full models
        for model_idx, model_name in enumerate(model_names):
            if model_name in self.trained_models:
                validation_predictions = self.trained_models[model_name].predict_proba(X_validation)[:, 1]
                meta_features_validation[:, model_idx] = validation_predictions
        
        # Train meta-learners
        meta_learner_models = {
            'logistic': LogisticRegression(random_state=RANDOM_SEED),
            'random_forest': RandomForestClassifier(
                n_estimators=50, max_depth=3, random_state=RANDOM_SEED
            ),
            'lightgbm': lgb.LGBMClassifier(
                objective='binary', num_leaves=7, learning_rate=0.1,
                n_estimators=50, random_state=RANDOM_SEED, verbose=-1
            )
        }
        
        best_meta_learner = None
        best_meta_score = 0
        meta_results = {}
        
        for meta_name, meta_model in meta_learner_models.items():
            try:
                # Train meta-learner
                meta_model.fit(meta_features_train, y_train)
                
                # Predict on validation meta-features
                meta_predictions = meta_model.predict_proba(meta_features_validation)[:, 1]
                meta_auc = roc_auc_score(y_validation, meta_predictions)
                
                meta_results[meta_name] = {
                    'model': meta_model,
                    'auc': meta_auc,
                    'predictions': meta_predictions
                }
                
                if meta_auc > best_meta_score:
                    best_meta_score = meta_auc
                    best_meta_learner = meta_model
                    best_meta_name = meta_name
                
                print(f"   {meta_name} meta-learner AUC: {meta_auc:.4f}")
                
            except Exception as e:
                print(f"   ❌ {meta_name} meta-learner failed: {str(e)[:30]}...")
        
        self.meta_learner = best_meta_learner
        return meta_results, best_meta_score, best_meta_name
    
    def create_weighted_ensemble(self, model_scores: dict, X_validation: np.ndarray,
                               y_validation: np.ndarray) -> tuple:
        """Create weighted ensemble based on model performance"""
        print("🎭 Creating weighted ensemble...")
        
        # Calculate weights based on validation AUC
        total_weight = 0
        self.model_weights = {}
        
        for model_name, score in model_scores.items():
            if model_name in self.trained_models:
                # Only positive weights for models better than random
                weight = max(0, score - 0.5)
                self.model_weights[model_name] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
        
        # Generate weighted ensemble predictions
        weighted_predictions = np.zeros(len(X_validation))
        for model_name, weight in self.model_weights.items():
            if model_name in self.trained_models and weight > 0:
                predictions = self.trained_models[model_name].predict_proba(X_validation)[:, 1]
                weighted_predictions += weight * predictions
        
        weighted_auc = roc_auc_score(y_validation, weighted_predictions)
        
        print(f"   Weighted ensemble AUC: {weighted_auc:.4f}")
        
        return weighted_predictions, weighted_auc
    
    def train_complete_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_validation: np.ndarray, y_validation: np.ndarray) -> dict:
        """Train complete ensemble system with all 4 components"""
        print("🎯 Training complete ensemble system...")
        print("=" * 60)
        
        # Component 1: Multiple model variants
        print("Component 1: Multiple Model Variants")
        print(f"✓ Created {len(self.base_models)} diverse algorithms")
        
        # Component 2: Cross-validation
        print("\nComponent 2: Cross-Validation")
        cv_results = self.perform_cross_validation(X_train, y_train)
        
        # Component 3: Bagging/Boosting training
        print("\nComponent 3: Bagging/Boosting Training")
        model_scores = self.train_base_models(X_train, y_train, X_validation, y_validation)
        
        # Component 4: Stacking meta-learner
        print("\nComponent 4: Stacking Meta-Learner")
        meta_results, best_meta_score, best_meta_name = self.create_stacking_meta_learner(
            X_train, y_train, X_validation, y_validation
        )
        
        # Weighted ensemble
        weighted_predictions, weighted_auc = self.create_weighted_ensemble(
            model_scores, X_validation, y_validation
        )
        
        # Summary results
        ensemble_results = {
            'individual_scores': model_scores,
            'cv_results': cv_results,
            'weighted_ensemble_auc': weighted_auc,
            'stacking_auc': best_meta_score,
            'best_meta_learner': best_meta_name,
            'model_weights': self.model_weights
        }
        
        return ensemble_results


class AdvancedEnsemble:
    """Production-ready ensemble predictor"""
    
    def __init__(self, trained_models: dict, model_weights: dict, 
                 meta_learner=None, approach: str = 'weighted'):
        self.trained_models = trained_models
        self.model_weights = model_weights
        self.meta_learner = meta_learner
        self.approach = approach
        self.model_names = list(trained_models.keys())
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions"""
        if self.approach == 'weighted':
            return self._weighted_prediction(X)
        elif self.approach == 'stacking' and self.meta_learner is not None:
            return self._stacking_prediction(X)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
    def _weighted_prediction(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted average predictions"""
        ensemble_predictions = np.zeros(len(X))
        
        for model_name, weight in self.model_weights.items():
            if model_name in self.trained_models and weight > 0:
                predictions = self.trained_models[model_name].predict_proba(X)[:, 1]
                ensemble_predictions += weight * predictions
        
        return np.column_stack([1 - ensemble_predictions, ensemble_predictions])
    
    def _stacking_prediction(self, X: np.ndarray) -> np.ndarray:
        """Generate stacking meta-learner predictions"""
        meta_features = np.zeros((len(X), len(self.trained_models)))
        
        for i, model_name in enumerate(self.model_names):
            if model_name in self.trained_models:
                predictions = self.trained_models[model_name].predict_proba(X)[:, 1]
                meta_features[:, i] = predictions
        
        return self.meta_learner.predict_proba(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)