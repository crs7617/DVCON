"""
Multi-Modal AI Fusion Accelerator Models Package

This package contains TensorFlow/TFLite model generators for:
- Vision Processing Unit (VPU) - CNN for threat detection
- Audio Processing Unit (APU) - RNN for sound classification  
- Motion Analysis Engine (MAE) - MLP for motion patterns
- Multi-Modal Fusion Engine - Attention-based fusion

Usage:
    from models.vision_model import generate_and_convert_vision
    from models.audio_model import generate_and_convert_audio
    from models.motion_model import generate_and_convert_motion
    from models.fusion_model import generate_and_convert_fusion
    
    # Generate individual models
    vision_model = generate_and_convert_vision()
    audio_model = generate_and_convert_audio() 
    motion_model = generate_and_convert_motion()
    fusion_model = generate_and_convert_fusion()
"""

__version__ = "1.0.0"
__author__ = "Multi-Modal AI Team"

# Import all model generators for easy access
from .vision_model import generate_and_convert_vision, create_simple_vision_model
from .audio_model import generate_and_convert_audio, create_simple_audio_model
from .motion_model import generate_and_convert_motion, create_simple_motion_model
from .fusion_model import generate_and_convert_fusion, create_fusion_model

# Define what gets imported with "from models import *"
__all__ = [
    'generate_and_convert_vision',
    'generate_and_convert_audio', 
    'generate_and_convert_motion',
    'generate_and_convert_fusion',
    'create_simple_vision_model',
    'create_simple_audio_model',
    'create_simple_motion_model',
    'create_fusion_model'
]

# ============================================================================