"""
Google Drive ML Pipeline Example
Run ML Challenge pipeline using images from Google Drive
"""

import os
from train_model_drive import train_and_predict_pipeline

def main():
    print("🚀 ML Challenge with Google Drive Images")
    print("=" * 50)
    
    # Check for credentials
    if not os.path.exists('credentials.json'):
        print("❌ ERROR: credentials.json not found!")
        print("   Please download your Google Drive API credentials")
        print("   and save as 'credentials.json' in the project root")
        return
    
    # Check for dataset
    if not os.path.exists('dataset/train.csv'):
        print("❌ ERROR: dataset/train.csv not found!")
        return
    
    if not os.path.exists('dataset/test.csv'):
        print("❌ ERROR: dataset/test.csv not found!")
        return
    
    print("✅ All prerequisites found")
    print("📁 Using Google Drive folder ID: 1ZXP3slTxtjvVaqTFrblfR8eR5lf07nNK")
    print("\n🔐 You will be prompted to authenticate with Google Drive...")
    
    # Run complete pipeline
    train_and_predict_pipeline()

if __name__ == "__main__":
    main()