#!/usr/bin/env python3
"""
Audio Model Generator for Multi-Modal AI Fusion Accelerator
Generates audio_model.tflite for sound classification using RNN
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_audio_model():
    """Lightweight RNN for audio threat detection - optimized for FPGA simulation"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 13)),  # 32 time steps, 13 MFCC features
        
        # Simple LSTM layers
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16),
        
        # Simple dense layers  
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: normal, scream, crash
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_and_convert_audio():
    print("Creating Simple Audio Model for Vivado Simulation...")
    
    # Create model
    model = create_simple_audio_model()
    model.summary()
    
    # Generate minimal dummy data
    x_dummy = np.random.random((50, 32, 13)).astype(np.float32)
    y_dummy = np.random.randint(3, size=(50,)).astype(np.int32)
    
    # Quick training
    print("Quick training for weight initialization...")
    model.fit(x_dummy, y_dummy, epochs=2, batch_size=10, verbose=1)
    
    # Convert to TFLite with quantization
    print("Converting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # INT8 quantization
    def representative_dataset():
        for _ in range(10):
            yield [np.random.random((1, 32, 13)).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'audio_model.tflite')
    
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Audio model saved: {filepath} ({len(tflite_model)} bytes)")
    return tflite_model

if __name__ == "__main__":
    generate_and_convert_audio()