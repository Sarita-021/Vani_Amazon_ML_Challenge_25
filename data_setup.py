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
    # NOTE: Since we only have sample_test.csv, we'll load it for demonstration.
    # In a real scenario, you'd also load 'train.csv'.
    TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'sample_test1.csv') 
    
    try:
        # Load the sample test data
        test_df = pd.read_csv(TEST_DATA_PATH)
        print(f"Successfully loaded {TEST_DATA_PATH}.")
    except FileNotFoundError:
        # If the data file is not in 'dataset/', check the root directory as a fallback
        TEST_DATA_PATH = 'sample_test1.csv'
        try:
            test_df = pd.read_csv(TEST_DATA_PATH)
            print(f"Successfully loaded {TEST_DATA_PATH}.")
            # If loaded from root, you might want to move it to the 'dataset/' folder later
        except FileNotFoundError:
            print(f"Error: Could not find '{TEST_DATA_PATH}' in either '{DATASET_FOLDER}/' or the root directory.")
            print("Please ensure your data files are in the correct location.")
            return

    # --- Initial EDA (Quick Check) ---
    print("\n--- Data Head (Sample Test) ---")
    print(test_df.head())
    print("\n--- Data Information (Sample Test) ---")
    test_df.info()
    
    # --- Image Downloading ---
    print(f"\nStarting image download to '{IMAGE_DOWNLOAD_FOLDER}'...")
    image_links = test_df['image_link'].tolist()
    
    # The download_images function uses multiprocessing, which is fast and efficient
    download_images(image_links, IMAGE_DOWNLOAD_FOLDER)
    
    print("\n✅ Image download complete!")
    print(f"Check the '{IMAGE_DOWNLOAD_FOLDER}' folder for the downloaded product images.")

if __name__ == "__main__":
    # Ensure you have moved 'sample_test1.csv' into a created 'dataset/' folder
    # or adjust the path variables if your data is elsewhere.
    setup_and_download_data()