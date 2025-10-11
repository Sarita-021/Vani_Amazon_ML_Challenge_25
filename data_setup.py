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
    # Define the folder where images will be saved
    IMAGE_DOWNLOAD_FOLDER = 'images'
    
    # Create the necessary directories if they don't exist
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_DOWNLOAD_FOLDER, exist_ok=True)
    
    # --- Data Loading ---
    print("Loading data...")
    TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'train.csv')
    TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'test.csv')
    
    try:
        # Load the training data for image downloading
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"Successfully loaded {TRAIN_DATA_PATH} with {len(train_df)} samples.")
        data_df = train_df
    except FileNotFoundError:
        try:
            # Fallback to test data if train not available
            test_df = pd.read_csv(TEST_DATA_PATH)
            print(f"Successfully loaded {TEST_DATA_PATH} with {len(test_df)} samples.")
            data_df = test_df
        except FileNotFoundError:
            print(f"Error: Could not find train.csv or test.csv in '{DATASET_FOLDER}/' folder.")
            print("Please ensure your data files are in the correct location.")
            return

    # --- Initial EDA (Quick Check) ---
    print("\n--- Data Head ---")
    print(data_df.head())
    print("\n--- Data Information ---")
    data_df.info()
    
    # --- Image Downloading ---
    print(f"\nStarting image download to '{IMAGE_DOWNLOAD_FOLDER}'...")
    image_links = data_df['image_link'].tolist()
    
    # The download_images function uses multiprocessing, which is fast and efficient
    download_images(image_links, IMAGE_DOWNLOAD_FOLDER)
    
    print("\n✅ Image download complete!")
    print(f"Check the '{IMAGE_DOWNLOAD_FOLDER}' folder for the downloaded product images.")

if __name__ == "__main__":
    # Ensure you have train.csv and test.csv in the 'dataset/' folder
    setup_and_download_data()