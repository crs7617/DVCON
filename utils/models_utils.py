#!/usr/bin/env python3
"""
Utility functions for Multi-Modal AI Fusion Accelerator
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import tensorflow as tf
        import numpy as np
        print(f"✅ TensorFlow {tf.__version__} found")
        print(f"✅ NumPy {np.__version__} found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install tensorflow numpy")
        return False

def setup_directories():
    """Create required project directories"""
    directories = [
        'output',
        'models', 
        'scripts',
        'utils',
        'docs',
        '.vscode'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
    # Create __init__.py files for Python packages
    for pkg_dir in ['models', 'utils']:
        init_file = Path(pkg_dir) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
    
    print("📁 Project directories created/verified")

def get_model_file_sizes():
    """Get sizes of generated model files"""
    output_dir = Path('output')
    model_files = [
        'vision_model.tflite',
        'audio_model.tflite', 
        'motion_model.tflite',
        'fusion_model.tflite'
    ]
    
    sizes = {}
    total_size = 0
    
    for model_file in model_files:
        model_path = output_dir / model_file
        if model_path.exists():
            size = model_path.stat().st_size
            sizes[model_file] = size
            total_size += size
        else:
            sizes[model_file] = 0
    
    return sizes, total_size

def print_project_structure():
    """Print the recommended project structure"""
    structure = """
📁 Recommended Project Structure:
multimodal-ai-fusion/
├── .vscode/
│   ├── settings.json
│   ├── launch.json
│   └── tasks.json
├── models/
│   ├── __init__.py
│   ├── vision_model.py
│   ├── audio_model.py
│   ├── motion_model.py
│   └── fusion_model.py
├── output/
│   ├── vision_model.tflite
│   ├── audio_model.tflite
│   ├── motion_model.tflite
│   └── fusion_model.tflite
├── scripts/
│   ├── generate_all_models.py
│   └── test_models.py
├── utils/
│   ├── __init__.py
│   └── model_utils.py
├── docs/
│   └── README.md
├── requirements.txt
└── main.py
"""
    print(structure)

def create_requirements_txt():
    """Create requirements.txt file"""
    requirements = """tensorflow>=2.10.0
numpy>=1.21.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ requirements.txt created")

def create_vscode_settings():
    """Create VS Code settings for the project"""
    vscode_dir = Path('.vscode')
    vscode_dir.mkdir(exist_ok=True)
    
    # settings.json
    settings = """{
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "files.associations": {
        "*.py": "python"
    },
    "python.analysis.extraPaths": [
        "./models",
        "./utils",
        "./scripts"
    ]
}"""
    
    with open(vscode_dir / 'settings.json', 'w') as f:
        f.write(settings)
    
    # launch.json
    launch = """{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Generate All Models",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": ["--generate-all"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Test All Models",
            "type": "python", 
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": ["--test-all"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Full Demo",
            "type": "python",
            "request": "launch", 
            "program": "${workspaceFolder}/main.py",
            "args": ["--demo"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}"""
    
    with open(vscode_dir / 'launch.json', 'w') as f:
        f.write(launch)
    
    print("✅ VS Code configuration created")

if __name__ == "__main__":
    print("Setting up Multi-Modal AI Fusion Accelerator project...")
    check_dependencies()
    setup_directories()
    create_requirements_txt()
    create_vscode_settings()
    print_project_structure()
    print("🎉 Project setup complete!")