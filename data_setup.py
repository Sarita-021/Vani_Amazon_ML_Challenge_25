import os
import pandas as pd
from src.utils import download_images

def setup_and_download_data():
    """
    Sets up the directory structure, loads the sample test data, 
    and downloads all product images for the provided links.
    """
    
    # --- Configuration ---
    # Define the folder where your CSV files are located
    DATASET_FOLDER = 'dataset'
    # Define the folders where images will be saved
    TRAIN_IMAGES_FOLDER = 'images/train'
    TEST_IMAGES_FOLDER = 'images/test'
    
    # Create the necessary directories if they don't exist
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
    
    # --- Data Loading ---
    print("Loading data...")
    TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'sample_train.csv')
    TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'sample_test.csv')
    
    # Load both training and test data
    try:
        # Load training data
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"Successfully loaded {TRAIN_DATA_PATH} with {len(train_df)} samples.")
        
        # Load test data
        test_df = pd.read_csv(TEST_DATA_PATH)
        print(f"Successfully loaded {TEST_DATA_PATH} with {len(test_df)} samples.")
        
        print(f"Total images to download: {len(train_df) + len(test_df)}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files: {e}")
        print("Please ensure both sample_train.csv and sample_test.csv are in the 'dataset/' folder.")
        return

    # --- Initial EDA (Quick Check) ---
    print("\n--- Training Data Head ---")
    print(train_df.head())
    print("\n--- Training Data Information ---")
    train_df.info()
    
    # --- Image Downloading ---
    print(f"\nDownloading training images to '{TRAIN_IMAGES_FOLDER}'...")
    train_links = train_df['image_link'].tolist()
    download_images(train_links, TRAIN_IMAGES_FOLDER)
    
    print(f"\nDownloading test images to '{TEST_IMAGES_FOLDER}'...")
    test_links = test_df['image_link'].tolist()
    download_images(test_links, TEST_IMAGES_FOLDER)
    
    print("\n✅ Image download complete!")
    print(f"Training images: '{TRAIN_IMAGES_FOLDER}'")
    print(f"Test images: '{TEST_IMAGES_FOLDER}'")

if __name__ == "__main__":
    # Downloads images for both train.csv and test.csv from 'dataset/' folder
    # This is required for the model to work on both training and test data
    setup_and_download_data()