"""
Utility Functions for Multi-Modal AI Fusion Accelerator

Common utilities for model conversion, validation, and helper functions.

Usage:
    from utils.model_utils import convert_to_tflite_int8, validate_tflite_model
"""

__version__ = "1.0.0"

# Import utility functions
from .model_utils import convert_to_tflite_int8, validate_tflite_model

__all__ = [
    'convert_to_tflite_int8',
    'validate_tflite_model'
]