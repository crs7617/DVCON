#!/usr/bin/env python3
"""
Master script to generate all TFLite models for Multi-Modal AI Fusion Accelerator
"""

import os
import sys
import time
from models.vision_model import generate_and_convert_vision
from models.audio_model import generate_and_convert_audio  
from models.motion_model import generate_and_convert_motion
from models.fusion_model import generate_and_convert_fusion

def main():
    print("=" * 60)
    print("🚀 MULTI-MODAL AI FUSION ACCELERATOR MODEL GENERATOR")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    start_time = time.time()
    total_size = 0
    
    try:
        # Generate all models
        print("1️⃣  Generating Vision Model...")
        vision_model = generate_and_convert_vision()
        total_size += len(vision_model)
        
        print("\n2️⃣  Generating Audio Model...")
        audio_model = generate_and_convert_audio()
        total_size += len(audio_model)
        
        print("\n3️⃣  Generating Motion Model...")
        motion_model = generate_and_convert_motion()
        total_size += len(motion_model)
        
        print("\n4️⃣  Generating Fusion Model...")
        fusion_model = generate_and_convert_fusion()
        total_size += len(fusion_model)
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 ALL MODELS GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Total model size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print(f"⏱️  Generation time: {elapsed:.1f} seconds")
        print("\n📁 Models saved in output/ directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()