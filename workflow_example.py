#!/usr/bin/env python3
"""
ML Challenge 2025 - Complete Workflow Example
============================================

This script demonstrates the proper ML workflow with separate training and prediction phases.

WORKFLOW:
1. Download images into separate train/test folders
2. Train model on training data and save learned components
3. Load trained model and predict on test data
4. Generate submission file

USAGE:
    python workflow_example.py
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} failed!")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False
    return True

def main():
    print("🎯 ML Challenge 2025 - Complete Workflow")
    print("=" * 50)
    
    # Step 1: Download images into separate folders
    print("\n📥 STEP 1: Download Images")
    if not run_command("python data_setup.py", "Downloading images into train/test folders"):
        return
    
    # Verify folder structure
    train_images = "images/train"
    test_images = "images/test"
    
    if os.path.exists(train_images) and os.path.exists(test_images):
        train_count = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
        test_count = len([f for f in os.listdir(test_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✅ Images organized: {train_count} training, {test_count} test images")
    else:
        print("❌ Image folders not found!")
        return
    
    # Step 2: Train model
    print("\n🧠 STEP 2: Train Model")
    if not run_command("python train_model.py train", "Training model and saving components"):
        return
    
    # Verify model files
    models_folder = "models"
    required_files = ["trained_model.pkl", "tfidf_vectorizer.pkl", "model_metadata.pkl"]
    
    if all(os.path.exists(os.path.join(models_folder, f)) for f in required_files):
        print("✅ All model components saved successfully!")
    else:
        print("❌ Some model components missing!")
        return
    
    # Step 3: Generate predictions
    print("\n🔮 STEP 3: Generate Predictions")
    if not run_command("python train_model.py predict", "Loading model and predicting on test data"):
        return
    
    # Verify output file
    output_file = "dataset/test_out.csv"
    if os.path.exists(output_file):
        print(f"✅ Predictions saved to {output_file}")
        
        # Show file info
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"📊 Output file contains {len(df)} predictions")
        print(f"💰 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"📋 Sample predictions:\n{df.head()}")
    else:
        print("❌ Output file not generated!")
        return
    
    print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📁 Files generated:")
    print(f"   • Training images: {train_images}/")
    print(f"   • Test images: {test_images}/")
    print(f"   • Trained model: {models_folder}/")
    print(f"   • Predictions: {output_file}")
    print("\n🚀 Ready for ML Challenge submission!")

if __name__ == "__main__":
    main()