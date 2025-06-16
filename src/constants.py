"""
Constants for Fraud Detection System
Contains all configuration values and hyperparameters
"""

# Model Architecture Constants
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_FUSION_DIM = 128
DEFAULT_HIDDEN_DIM = 64
DEFAULT_LATENT_DIM = 32
DEFAULT_NUM_ATTENTION_HEADS = 4
DEFAULT_DROPOUT_RATE = 0.3

# Training Constants
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 30
DEFAULT_PATIENCE = 10
DEFAULT_GRADIENT_CLIP_VAL = 1.0

# Data Processing Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
SMOTE_K_NEIGHBORS = 3

# Focal Loss Constants
DEFAULT_FOCAL_ALPHA = 1.0
DEFAULT_FOCAL_GAMMA = 2.0

# Ensemble Constants
ENSEMBLE_N_ESTIMATORS = 100
ENSEMBLE_MAX_DEPTH = 8
ENSEMBLE_CV_FOLDS = 5
ENSEMBLE_META_LEARNER_MODELS = ['logistic', 'random_forest', 'lightgbm']

# File Paths
DATA_FILE_PATH = "/home/nb3283/fraud_detection_system/data/data.xlsx"
MODEL_SAVE_PATH = "/home/nb3283/fraud_detection_system/models/"
CONFIG_PATH = "/home/nb3283/fraud_detection_system/config/"

# Model Performance Thresholds
MIN_AUC_THRESHOLD = 0.7
TARGET_AUC_THRESHOLD = 0.85
MAX_FALSE_POSITIVE_RATE = 0.15

# Feature Engineering Constants
AGE_BINS = [0, 25, 35, 50, 65, 100]
AGE_LABELS = ['young', 'adult', 'middle', 'senior', 'elderly']
QUANTILE_BINS = 5
QUANTILE_LABELS = ['low', 'med_low', 'med', 'med_high', 'high']

# Business Constants
FRAUD_INVESTIGATION_COST = 1000
FALSE_POSITIVE_COST = 100
FALSE_NEGATIVE_COST = 5000