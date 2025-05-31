#!/usr/bin/env python3
"""
Fusion Model Generator for Multi-Modal AI Fusion Accelerator
Generates fusion_model.tflite (TensorFlow Lite model) for multi-modal threat detection
"""

import tensorflow as tf
import numpy as np
import os

def create_fusion_model():
    """Simple fusion model that combines outputs from vision, audio, and motion models"""
    
    # Input from each modality (outputs of individual models)
    vision_input = tf.keras.layers.Input(shape=(1,), name='vision_conf')      # Vision confidence
    audio_input = tf.keras.layers.Input(shape=(3,), name='audio_probs')       # Audio probabilities  
    motion_input = tf.keras.layers.Input(shape=(4,), name='motion_probs')     # Motion probabilities
    
    # Simple attention mechanism (learnable weights)
    vision_weighted = tf.keras.layers.Dense(4, activation='tanh')(vision_input)
    audio_weighted = tf.keras.layers.Dense(4, activation='tanh')(audio_input)
    motion_weighted = tf.keras.layers.Dense(4, activation='tanh')(motion_input)
    
    # Replace concatenation with element-wise addition
    fused = tf.keras.layers.Add()([vision_weighted, audio_weighted, motion_weighted])
    
    # Fusion processing
    x = tf.keras.layers.Dense(16, activation='relu')(fused)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    
    # Final threat confidence score
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='threat_confidence')(x)
    
    model = tf.keras.Model(
        inputs=[vision_input, audio_input, motion_input],
        outputs=output
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Binary cross-entropy loss for classification
    return model

def generate_and_convert_fusion():
    print("Creating Multi-Modal Fusion Model for FPGA Simulation...")
    
    # Create model
    model = create_fusion_model()
    model.summary()
    
    # Generate dummy data for fusion training
    vision_dummy = np.random.random((100, 1)).astype(np.float32)
    audio_dummy = np.random.random((100, 3)).astype(np.float32)
    motion_dummy = np.random.random((100, 4)).astype(np.float32)
    threat_labels = np.random.randint(2, size=(100, 1)).astype(np.float32)
    
    # Quick training
    print("Quick training for weight initialization...")
    model.fit(
        [vision_dummy, audio_dummy, motion_dummy], 
        threat_labels, 
        epochs=3, 
        batch_size=20, 
        verbose=1
    )
    
    # Convert to TFLite with quantization
    print("Converting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # INT8 quantization
    def representative_dataset():
        for _ in range(20):
            yield [
                np.random.random((1, 1)).astype(np.float32),
                np.random.random((1, 3)).astype(np.float32), 
                np.random.random((1, 4)).astype(np.float32)
            ]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # INT8 quantization for TensorFlow Lite
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()  # Convert the model to TensorFlow Lite format
    
    # Save TFLite model
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'fusion_model.tflite')  # Save the TensorFlow Lite model
    
    with open(filepath, 'wb') as f:
        f.write(tflite_model)  # Write the TensorFlow Lite model to a file
    
    print(f"✅ Fusion model saved: {filepath} ({len(tflite_model)} bytes)")  # Confirm model save
    return tflite_model  # Return the TensorFlow Lite model

if __name__ == "__main__":
    generate_and_convert_fusion()