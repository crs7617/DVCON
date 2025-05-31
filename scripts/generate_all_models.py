#!/usr/bin/env python3
"""
Generate all TFLite models for Multi-Modal AI Fusion Accelerator
Vivado simulation and power analysis

Run this single script to generate all required .tflite files:
- vision_model.tflite
- audio_model.tflite  
- motion_model.tflite
- fusion_model.tflite

No virtual environments needed - just install tensorflow and run!
"""

import tensorflow as tf
import numpy as np
import os
import time

def create_vision_model():
    """Vision Processing Unit - CNN for threat detection"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_audio_model():
    """Audio Processing Unit - RNN for sound classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 13)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_motion_model():
    """Motion Analysis Engine - MLP for motion patterns"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_fusion_model():
    """Multi-Modal Fusion Engine - Attention-based fusion"""
    vision_input = tf.keras.layers.Input(shape=(1,), name='vision')
    audio_input = tf.keras.layers.Input(shape=(3,), name='audio')
    motion_input = tf.keras.layers.Input(shape=(4,), name='motion')
    
    vision_weighted = tf.keras.layers.Dense(4, activation='tanh')(vision_input)
    audio_weighted = tf.keras.layers.Dense(4, activation='tanh')(audio_input)
    motion_weighted = tf.keras.layers.Dense(4, activation='tanh')(motion_input)
    
    fused = tf.keras.layers.Concatenate()([vision_weighted, audio_weighted, motion_weighted])
    x = tf.keras.layers.Dense(16, activation='relu')(fused)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[vision_input, audio_input, motion_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def convert_to_tflite(model, model_name, input_shapes):
    """Convert Keras model to quantized TFLite"""
    print(f"Converting {model_name} to TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Create representative dataset for quantization
    def representative_dataset():
        for _ in range(10):
            if isinstance(input_shapes[0], tuple):  # Multi-input model
                yield [np.random.random((1,) + shape).astype(np.float32) for shape in input_shapes]
            else:  # Single input model
                yield [np.random.random((1,) + input_shapes).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    filename = f"{model_name}.tflite"
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ {filename} saved ({len(tflite_model)} bytes)")
    return len(tflite_model)

def main():
    print("=" * 60)
    print("🚀 MULTI-MODAL AI FUSION ACCELERATOR MODEL GENERATOR")
    print("=" * 60)
    print("Generating TFLite models for Vivado simulation...")
    print()
    
    total_size = 0
    start_time = time.time()
    
    # 1. Vision Model
    print("1️⃣  Creating Vision Processing Unit (VPU)...")
    vision_model = create_vision_model()
    x_vis = np.random.random((30, 64, 64, 3)).astype(np.float32)
    y_vis = np.random.randint(2, size=(30, 1)).astype(np.float32)
    vision_model.fit(x_vis, y_vis, epochs=2, batch_size=10, verbose=0)
    total_size += convert_to_tflite(vision_model, "vision_model", (64, 64, 3))
    print()
    
    # 2. Audio Model  
    print("2️⃣  Creating Audio Processing Unit (APU)...")
    audio_model = create_audio_model()
    x_aud = np.random.random((30, 32, 13)).astype(np.float32)
    y_aud = np.random.randint(3, size=(30,)).astype(np.int32)
    audio_model.fit(x_aud, y_aud, epochs=2, batch_size=10, verbose=0)
    total_size += convert_to_tflite(audio_model, "audio_model", (32, 13))
    print()
    
    # 3. Motion Model
    print("3️⃣  Creating Motion Analysis Engine (MAE)...")
    motion_model = create_motion_model()
    x_mot = np.random.random((50, 6)).astype(np.float32)
    y_mot = np.random.randint(4, size=(50,)).astype(np.int32)
    motion_model.fit(x_mot, y_mot, epochs=2, batch_size=15, verbose=0)
    total_size += convert_to_tflite(motion_model, "motion_model", (6,))
    print()
    
    # 4. Fusion Model
    print("4️⃣  Creating Multi-Modal Fusion Engine...")
    fusion_model = create_fusion_model()
    x_vis_f = np.random.random((40, 1)).astype(np.float32)
    x_aud_f = np.random.random((40, 3)).astype(np.float32)
    x_mot_f = np.random.random((40, 4)).astype(np.float32)
    y_fusion = np.random.randint(2, size=(40, 1)).astype(np.float32)
    fusion_model.fit([x_vis_f, x_aud_f, x_mot_f], y_fusion, epochs=2, batch_size=10, verbose=0)
    total_size += convert_to_tflite(fusion_model, "fusion_model", [(1,), (3,), (4,)])
    print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 60)
    print("🎉 ALL MODELS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Files created in current directory:")
    print(f"   • vision_model.tflite  - Vision Processing Unit")
    print(f"   • audio_model.tflite   - Audio Processing Unit")
    print(f"   • motion_model.tflite  - Motion Analysis Engine")
    print(f"   • fusion_model.tflite  - Multi-Modal Fusion Engine")
    print()
    print(f"📊 Total model size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"⏱️  Generation time: {elapsed:.1f} seconds")
    print()
    print("🔄 TO SHARE WITH HARDWARE TEAM:")
    print("   1. Copy all .tflite files to shared folder")
    print("   2. These are INT8 quantized for low power consumption")
    print("   3. Use in Vivado for power analysis and simulation")
    print("=" * 60)

if __name__ == "__main__":
    main()