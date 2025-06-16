"""
Fraud Detection System Package
A comprehensive multi-modal fraud detection system with ensemble methods
"""

from .constants import *
from .data_processor import DataProcessor
from .models import *
from .ensemble import EnsembleSystem, AdvancedEnsemble
from .trainer import FraudDetectionTrainer

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"