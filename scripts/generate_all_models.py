#!/usr/bin/env python3
"""
FIXED: Generate all TFLite models for Multi-Modal AI Fusion Accelerator
Vivado simulation and power analysis

Fixed concatenation dimension mismatch in fusion model!
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
    """Audio Processing Unit - 1D CNN for sound classification (FIXED - no LSTM)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 13)),  # 32 time steps, 13 MFCC features
        
        # Use 1D convolutions instead of LSTM for TFLite compatibility
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalMaxPooling1D(),
        
        # Dense layers for classification
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: normal, scream, crash
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
    """SIMPLIFIED Multi-Modal Fusion Engine - Avoid concatenation issues"""
    vision_input = tf.keras.layers.Input(shape=(1,), name='vision')
    audio_input = tf.keras.layers.Input(shape=(3,), name='audio')  
    motion_input = tf.keras.layers.Input(shape=(4,), name='motion')
    
    # Process each modality separately
    vision_features = tf.keras.layers.Dense(8, activation='relu')(vision_input)
    audio_features = tf.keras.layers.Dense(8, activation='relu')(audio_input)
    motion_features = tf.keras.layers.Dense(8, activation='relu')(motion_input)
    
    # Instead of concatenation, use element-wise operations
    # Add all features together (fusion by addition)
    fused = tf.keras.layers.Add()([vision_features, audio_features, motion_features])
    
    # Process fused features
    x = tf.keras.layers.Dense(16, activation='relu')(fused)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[vision_input, audio_input, motion_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def convert_to_tflite_robust(model, model_name, input_shapes):
    """Convert Keras model to quantized TFLite with robust error handling"""
    print(f"Converting {model_name} to TFLite...")
    
    # Create representative dataset for quantization
    def representative_dataset():
        for _ in range(10):
            if isinstance(input_shapes[0], tuple):  # Multi-input model
                yield [np.random.random((1,) + shape).astype(np.float32) for shape in input_shapes]
            else:  # Single input model
                yield [np.random.random((1,) + input_shapes).astype(np.float32)]
    
    # Try INT8 quantization first
    try:
        print(f"   Attempting INT8 quantization for {model_name}...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.allow_custom_ops = False
        converter.experimental_new_converter = True
        
        tflite_model = converter.convert()
        quantization_type = "INT8"
        
    except Exception as e:
        print(f"   ⚠️  INT8 failed for {model_name}, trying float16...")
        print(f"   Error: {str(e)[:100]}...")
        
        try:
            # Fallback to float16
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            quantization_type = "FLOAT16"
            
        except Exception as e2:
            print(f"   ⚠️  Float16 failed for {model_name}, using default optimization...")
            print(f"   Error: {str(e2)[:100]}...")
            
            # Final fallback - basic optimization only
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            quantization_type = "DEFAULT"
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    filename = f"output/{model_name}.tflite"
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"   ✅ {filename} saved ({size_kb:.1f} KB, {quantization_type})")
    return len(tflite_model)

def main():
    print("=" * 70)
    print("🚀 FIXED: MULTI-MODAL AI FUSION ACCELERATOR MODEL GENERATOR")
    print("=" * 70)
    print("Generating TFLite models for Vivado simulation...")
    print("🔧 Fixed fusion model using element-wise addition instead of concatenation")
    print()
    
    total_size = 0
    start_time = time.time()
    
    # 1. Vision Model
    print("1️⃣  Creating Vision Processing Unit (VPU)...")
    vision_model = create_vision_model()
    x_vis = np.random.random((30, 64, 64, 3)).astype(np.float32)
    y_vis = np.random.randint(2, size=(30, 1)).astype(np.float32)
    vision_model.fit(x_vis, y_vis, epochs=2, batch_size=10, verbose=0)
    total_size += convert_to_tflite_robust(vision_model, "vision_model", (64, 64, 3))
    print()
    
    # 2. Audio Model (FIXED)
    print("2️⃣  Creating Audio Processing Unit (APU) - FIXED...")
    audio_model = create_audio_model()
    x_aud = np.random.random((30, 32, 13)).astype(np.float32)
    y_aud = np.random.randint(3, size=(30,)).astype(np.int32)
    audio_model.fit(x_aud, y_aud, epochs=2, batch_size=10, verbose=0)
    total_size += convert_to_tflite_robust(audio_model, "audio_model", (32, 13))
    print()
    
    # 3. Motion Model
    print("3️⃣  Creating Motion Analysis Engine (MAE)...")
    motion_model = create_motion_model()
    x_mot = np.random.random((50, 6)).astype(np.float32)
    y_mot = np.random.randint(4, size=(50,)).astype(np.int32)
    motion_model.fit(x_mot, y_mot, epochs=2, batch_size=15, verbose=0)
    total_size += convert_to_tflite_robust(motion_model, "motion_model", (6,))
    print()
    
    # 4. Fusion Model (FIXED)
    print("4️⃣  Creating Multi-Modal Fusion Engine (FIXED)...")
    fusion_model = create_fusion_model()
    
    # Print model summary to verify architecture
    print("   Fusion model architecture:")
    fusion_model.summary()
    
    # Train fusion model
    x_vis_f = np.random.random((40, 1)).astype(np.float32)
    x_aud_f = np.random.random((40, 3)).astype(np.float32)
    x_mot_f = np.random.random((40, 4)).astype(np.float32)
    y_fusion = np.random.randint(2, size=(40, 1)).astype(np.float32)
    fusion_model.fit([x_vis_f, x_aud_f, x_mot_f], y_fusion, epochs=2, batch_size=10, verbose=0)

    # Convert to TFLite
    total_size += convert_to_tflite_robust(fusion_model, "fusion_model", [(1,), (3,), (4,)])
    print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 70)
    print("🎉 ALL MODELS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📁 Files created in output/ directory:")
    print(f"   • vision_model.tflite  - Vision Processing Unit")
    print(f"   • audio_model.tflite   - Audio Processing Unit (FIXED)")
    print(f"   • motion_model.tflite  - Motion Analysis Engine")
    print(f"   • fusion_model.tflite  - Multi-Modal Fusion Engine (FIXED)")
    print()
    print(f"📊 Total model size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"⏱️  Generation time: {elapsed:.1f} seconds")
    print()
    print("🔄 READY FOR HARDWARE TEAM:")
    print("   1. All models are TFLite compatible")
    print("   2. Quantized for low power consumption")
    print("   3. No LSTM compatibility issues")
    print("   4. Fixed fusion model using element-wise addition")
    print("   5. Use in Vivado for power analysis and simulation")
    print("=" * 70)

if __name__ == "__main__":
    main()