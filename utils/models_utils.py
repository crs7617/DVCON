"""
Common utilities for model conversion and validation
"""

import tensorflow as tf
import numpy as np
import os

def convert_to_tflite_int8(model, model_name, input_shapes, output_dir="output"):
    """
    Convert Keras model to INT8 quantized TFLite
    """
    print(f"Converting {model_name} to TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for quantization
    def representative_dataset():
        for _ in range(10):
            if isinstance(input_shapes[0], tuple):  # Multi-input
                yield [np.random.random((1,) + shape).astype(np.float32) for shape in input_shapes]
            else:  # Single input
                yield [np.random.random((1,) + input_shapes).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{model_name}.tflite")
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ {filepath} saved ({len(tflite_model)} bytes)")
    return tflite_model

def validate_tflite_model(tflite_path):
    """
    Validate TFLite model can be loaded and interpreted
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        print(f"✅ {tflite_path} validated successfully")
        return True
    except Exception as e:
        print(f"❌ {tflite_path} validation failed: {e}")
        return False