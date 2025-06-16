"""
Training Module for Fraud Detection System
Handles training orchestration for both neural networks and ensemble models
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import numpy as np
from .models import MultiModalFraudDetector, AnomalyAutoencoder
from .ensemble import EnsembleSystem, AdvancedEnsemble
from .constants import *


class TabularDataset(Dataset):
    """Dataset for tabular fraud detection data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        # Convert to dense array if sparse
        if hasattr(features, "toarray"):
            features = features.toarray()
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index: int) -> dict:
        return {
            'tab': self.features[index],
            'label': self.labels[index]
        }


class FraudDetectionTrainer:
    """Main training orchestrator"""
    
    def __init__(self):
        self.neural_model = None
        self.ensemble_system = None
        self.trained_autoencoder = None
    
    def train_autoencoder(self, normal_data: np.ndarray, input_dim: int) -> AnomalyAutoencoder:
        """Train anomaly detection autoencoder on normal data only"""
        print("🔍 Training anomaly detection autoencoder...")
        
        autoencoder = AnomalyAutoencoder(input_dim)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=DEFAULT_LEARNING_RATE)
        autoencoder.train()
        
        normal_tensor = torch.tensor(normal_data, dtype=torch.float32)
        
        for epoch in range(50):  # Fixed number of epochs for autoencoder
            total_loss = 0
            batch_count = 0
            
            for i in range(0, len(normal_tensor), DEFAULT_BATCH_SIZE):
                batch = normal_tensor[i:i + DEFAULT_BATCH_SIZE]
                
                # Skip small batches to avoid BatchNorm issues
                if len(batch) <= 1:
                    continue
                
                optimizer.zero_grad()
                reconstructed, _ = autoencoder(batch)
                loss = torch.nn.functional.mse_loss(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            if (epoch + 1) % 10 == 0 and batch_count > 0:
                average_loss = total_loss / batch_count
                print(f"   Epoch {epoch + 1}/50, Average Loss: {average_loss:.6f}")
        
        autoencoder.eval()
        self.trained_autoencoder = autoencoder
        print("   ✅ Autoencoder training completed")
        
        return autoencoder
    
    def train_neural_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_validation: np.ndarray, y_validation: np.ndarray,
                          use_focal_loss: bool = True) -> MultiModalFraudDetector:
        """Train the multi-modal neural network"""
        print("🧠 Training multi-modal neural network...")
        
        # Set tensor precision for optimal performance
        torch.set_float32_matmul_precision('medium')
        
        # Create datasets and data loaders
        train_dataset = TabularDataset(X_train, y_train)
        validation_dataset = TabularDataset(X_validation, y_validation)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=DEFAULT_BATCH_SIZE, 
            shuffle=True, 
            num_workers=0  # Set to 0 for Docker compatibility
        )
        validation_dataloader = DataLoader(
            validation_dataset, 
            batch_size=DEFAULT_BATCH_SIZE, 
            shuffle=False, 
            num_workers=0
        )
        
        # Train autoencoder on normal data
        normal_mask = y_train == 0
        normal_data = X_train[normal_mask]
        trained_autoencoder = self.train_autoencoder(normal_data, X_train.shape[1])
        
        # Initialize multi-modal model
        model = MultiModalFraudDetector(
            tabular_input_dim=X_train.shape[1],
            use_focal_loss=use_focal_loss
        )
        
        # Replace model's autoencoder with trained one
        model.anomaly_autoencoder = trained_autoencoder
        
        # Configure callbacks
        callbacks = [
            EarlyStopping(
                monitor='validation/auc',
                mode='max',
                patience=DEFAULT_PATIENCE,
                verbose=True
            ),
            ModelCheckpoint(
                monitor='validation/auc',
                mode='max',
                save_top_k=1,
                verbose=True,
                dirpath=MODEL_SAVE_PATH,
                filename='best_multimodal_fraud_detector'
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Configure trainer
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            max_epochs=DEFAULT_MAX_EPOCHS,
            callbacks=callbacks,
            benchmark=True,
            log_every_n_steps=5,
            gradient_clip_val=DEFAULT_GRADIENT_CLIP_VAL,
            accumulate_grad_batches=1,
            enable_progress_bar=True
        )
        
        # Train the model
        print("   🚀 Starting neural network training...")
        trainer.fit(model, train_dataloader, validation_dataloader)
        
        # Load best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            model = MultiModalFraudDetector.load_from_checkpoint(best_model_path)
            print(f"   ✅ Best model loaded from {best_model_path}")
        
        self.neural_model = model
        print("   ✅ Neural network training completed")
        
        return model
    
    def train_ensemble_system(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_validation: np.ndarray, y_validation: np.ndarray) -> AdvancedEnsemble:
        """Train the complete ensemble system"""
        print("🎯 Training ensemble system...")
        
        # Convert sparse matrices to dense if needed
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
            X_validation = X_validation.toarray()
        
        # Initialize and train ensemble system
        self.ensemble_system = EnsembleSystem()
        
        # Train complete ensemble with all 4 components
        ensemble_results = self.ensemble_system.train_complete_ensemble(
            X_train, y_train, X_validation, y_validation
        )
        
        # Create final ensemble predictor
        best_ensemble_approach = self._determine_best_ensemble_approach(ensemble_results)
        
        final_ensemble = AdvancedEnsemble(
            trained_models=self.ensemble_system.trained_models,
            model_weights=self.ensemble_system.model_weights,
            meta_learner=self.ensemble_system.meta_learner,
            approach=best_ensemble_approach
        )
        
        print("   ✅ Ensemble system training completed")
        
        return final_ensemble, ensemble_results
    
    def _determine_best_ensemble_approach(self, ensemble_results: dict) -> str:
        """Determine the best ensemble approach based on performance"""
        weighted_auc = ensemble_results.get('weighted_ensemble_auc', 0)
        stacking_auc = ensemble_results.get('stacking_auc', 0)
        
        if stacking_auc > weighted_auc:
            return 'stacking'
        else:
            return 'weighted'
    
    def evaluate_model_performance(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance on test data"""
        from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
        
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            # Ensemble model
            test_probabilities = model.predict_proba(X_test)[:, 1]
            test_predictions = model.predict(X_test)
        else:
            # Neural network model
            model.eval()
            test_dataset = TabularDataset(X_test, y_test)
            test_dataloader = DataLoader(test_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
            
            all_probabilities = []
            all_predictions = []
            device = next(model.parameters()).device
            
            with torch.no_grad():
                for batch in test_dataloader:
                    # Move batch to same device as model
                    batch_data = batch['tab'].to(device)
                    logits, _ = model(batch_data)
                    probabilities = torch.sigmoid(logits)
                    predictions = probabilities > 0.5
                    
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
            
            test_probabilities = np.array(all_probabilities)
            test_predictions = np.array(all_predictions)
        
        # Calculate metrics
        test_auc = roc_auc_score(y_test, test_probabilities)
        confusion_mat = confusion_matrix(y_test, test_predictions)
        classification_rep = classification_report(y_test, test_predictions, output_dict=True)
        
        # Extract confusion matrix components
        tn, fp, fn, tp = confusion_mat.ravel()
        
        # Calculate business metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        
        performance_metrics = {
            'auc': test_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'classification_report': classification_rep
        }
        
        return performance_metrics
    
    def save_models(self, neural_model=None, ensemble_model=None) -> None:
        """Save trained models"""
        import joblib
        import os
        
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        if neural_model:
            torch.save(neural_model.state_dict(), f"{MODEL_SAVE_PATH}/neural_fraud_detector.pth")
            print(f"   💾 Neural model saved to {MODEL_SAVE_PATH}/neural_fraud_detector.pth")
        
        if ensemble_model:
            joblib.dump(ensemble_model, f"{MODEL_SAVE_PATH}/ensemble_fraud_detector.pkl")
            print(f"   💾 Ensemble model saved to {MODEL_SAVE_PATH}/ensemble_fraud_detector.pkl")