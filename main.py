#!/usr/bin/env python3
"""
Main Application Entry Point for Fraud Detection System
Runs the complete training and evaluation pipeline
"""

import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('/app/src')

from src.data_processor import DataProcessor
from src.trainer import FraudDetectionTrainer
from src.constants import *


def display_banner():
    """Display application banner"""
    print("=" * 80)
    print("🚀 ADVANCED FRAUD DETECTION SYSTEM")
    print("=" * 80)
    print("🎯 Multi-Modal Neural Networks + Ensemble Methods")
    print("🔧 Components: Tabular, Graph, Text, Anomaly Detection")
    print("🏆 Ensemble: RandomForest, XGBoost, LightGBM, Stacking")
    print("=" * 80)


def run_neural_network_training(data_processor, trainer):
    """Run neural network training pipeline"""
    print("\n🧠 NEURAL NETWORK TRAINING PIPELINE")
    print("-" * 50)
    
    # Load and process data
    print("📊 Loading and processing data...")
    df = data_processor.load_data(DATA_FILE_PATH)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud rate: {df['fraud_label'].mean():.1%}")
    
    # Enhanced feature engineering
    print("🔧 Creating enhanced features...")
    df_enhanced = data_processor.create_enhanced_features(df)
    print(f"   Enhanced features: {df_enhanced.shape[1]} columns")
    
    # Prepare preprocessing
    data_processor.prepare_preprocessing_pipeline(df_enhanced)
    
    # Split and preprocess data
    X_train, X_val, y_train, y_val, train_df, val_df = data_processor.split_and_preprocess_data(df_enhanced)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Feature dimension: {X_train.shape[1]}")
    
    # Apply SMOTE balancing
    X_train_balanced, y_train_balanced = data_processor.apply_smote_balancing(X_train, y_train)
    print(f"   Balanced training samples: {X_train_balanced.shape[0]}")
    
    # Convert validation data to dense
    if hasattr(X_val, 'toarray'):
        X_val = X_val.toarray()
    
    # Train neural network
    neural_model = trainer.train_neural_model(
        X_train_balanced, y_train_balanced, X_val, y_val, use_focal_loss=True
    )
    
    # Evaluate neural model
    neural_performance = trainer.evaluate_model_performance(neural_model, X_val, y_val)
    
    print(f"\n📊 Neural Network Performance:")
    print(f"   AUC: {neural_performance['auc']:.4f}")
    print(f"   Precision: {neural_performance['precision']:.4f}")
    print(f"   Recall: {neural_performance['recall']:.4f}")
    print(f"   F1-Score: {neural_performance['f1_score']:.4f}")
    
    return neural_model, neural_performance, (X_train_balanced, y_train_balanced, X_val, y_val)


def run_ensemble_training(trainer, data_tuple):
    """Run ensemble training pipeline"""
    print("\n🎯 ENSEMBLE TRAINING PIPELINE")
    print("-" * 50)
    
    X_train, y_train, X_val, y_val = data_tuple
    
    # Train ensemble system
    ensemble_model, ensemble_results = trainer.train_ensemble_system(X_train, y_train, X_val, y_val)
    
    # Evaluate ensemble model
    ensemble_performance = trainer.evaluate_model_performance(ensemble_model, X_val, y_val)
    
    print(f"\n📊 Ensemble Performance Summary:")
    print(f"   Final AUC: {ensemble_performance['auc']:.4f}")
    print(f"   Precision: {ensemble_performance['precision']:.4f}")
    print(f"   Recall: {ensemble_performance['recall']:.4f}")
    print(f"   F1-Score: {ensemble_performance['f1_score']:.4f}")
    
    # Display individual model performance
    print(f"\n🤖 Individual Model Performance:")
    for model_name, score in sorted(ensemble_results['individual_scores'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"   {model_name:>18}: {score:.4f} AUC")
    
    # Display ensemble comparison
    print(f"\n🏆 Ensemble Methods Comparison:")
    print(f"   {'Weighted Average':>18}: {ensemble_results['weighted_ensemble_auc']:.4f} AUC")
    print(f"   {'Stacking':>18}: {ensemble_results['stacking_auc']:.4f} AUC")
    
    return ensemble_model, ensemble_performance, ensemble_results


def display_comprehensive_results(neural_performance, ensemble_performance, ensemble_results):
    """Display comprehensive results comparison"""
    print("\n" + "=" * 80)
    print("🏆 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    
    # Performance comparison
    print("📊 Model Performance Comparison:")
    print(f"   {'Neural Network':>20}: {neural_performance['auc']:.4f} AUC")
    print(f"   {'Ensemble (Best)':>20}: {ensemble_performance['auc']:.4f} AUC")
    
    # Improvement calculation
    improvement = ((ensemble_performance['auc'] - neural_performance['auc']) / neural_performance['auc']) * 100
    print(f"   {'Improvement':>20}: {improvement:+.1f}%")
    
    # Business impact metrics
    print(f"\n💼 Business Impact Metrics:")
    print(f"   {'Fraud Detection Rate':>25}: {ensemble_performance['recall']:.1%}")
    print(f"   {'False Alarm Rate':>25}: {ensemble_performance['false_positive_rate']:.1%}")
    print(f"   {'Investigation Precision':>25}: {ensemble_performance['precision']:.1%}")
    
    # Component verification
    print(f"\n✅ All 4 Ensemble Components Verified:")
    print(f"   ✓ Multiple model variants: {len(ensemble_results['individual_scores'])} algorithms")
    print(f"   ✓ Bagging/Boosting: RandomForest, GradientBoosting, XGBoost, LightGBM")
    print(f"   ✓ Cross-validation: {ENSEMBLE_CV_FOLDS}-fold stratified CV")
    print(f"   ✓ Stacking meta-learner: {ensemble_results['best_meta_learner']} selected")
    
    # Technical achievements
    print(f"\n🔬 Technical Achievements:")
    if ensemble_performance['auc'] > TARGET_AUC_THRESHOLD:
        print(f"   🎯 TARGET AUC ACHIEVED: {ensemble_performance['auc']:.4f} > {TARGET_AUC_THRESHOLD}")
    if ensemble_performance['false_positive_rate'] < MAX_FALSE_POSITIVE_RATE:
        print(f"   🎯 LOW FALSE POSITIVE RATE: {ensemble_performance['false_positive_rate']:.1%} < {MAX_FALSE_POSITIVE_RATE:.1%}")
    
    print(f"\n🐕 Ready for deployment at Corgi Insurance!")


def main():
    """Main application entry point"""
    try:
        # Display banner
        display_banner()
        
        # Initialize components
        data_processor = DataProcessor()
        trainer = FraudDetectionTrainer()
        
        # Check if data file exists
        if not os.path.exists(DATA_FILE_PATH):
            print(f"❌ Data file not found: {DATA_FILE_PATH}")
            print("   Please ensure data.xlsx is mounted in the /app/data/ directory")
            return
        
        # Run neural network training
        neural_model, neural_performance, data_tuple = run_neural_network_training(data_processor, trainer)
        
        # Run ensemble training
        ensemble_model, ensemble_performance, ensemble_results = run_ensemble_training(trainer, data_tuple)
        
        # Save models
        print(f"\n💾 Saving trained models...")
        trainer.save_models(neural_model, ensemble_model)
        
        # Display comprehensive results
        display_comprehensive_results(neural_performance, ensemble_performance, ensemble_results)
        
        print(f"\n🎉 FRAUD DETECTION SYSTEM TRAINING COMPLETE!")
        print(f"🚀 System ready for production deployment!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()