import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import QuantileTransformer

# Import your feature extraction modules
from src.text_features import engineer_text_features
from src.image_features import extract_comprehensive_image_features # Assuming the final function name from your latest upload

# --- Configuration ---
DATASET_FOLDER = 'dataset'
TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'train.csv') # Assuming you have this file
TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'test.csv')   # Assuming you have this file
OUTPUT_PATH = os.path.join(DATASET_FOLDER, 'predictions.csv')

# --- SMAPE Metric Function ---
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    # Clip predictions to prevent division by zero or log(0) issues
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def train_and_predict_pipeline():
    # --- 1. Data Loading ---
    print("🚀 Loading Training and Test Data...")
    try:
        # NOTE: Using 'sample_test1.csv' and 'sample_code.py' structure for now
        # You must change this to 'train.csv' and 'test.csv' for submission
        train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test1.csv')) # Replace with 'train.csv'
        test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test1.csv'))  # Replace with 'test.csv'
        
        # In a real scenario, you'd use the actual price column from train.csv
        # For this demo with sample_test1, we'll create a dummy target for structure
        if 'price' not in train_df.columns:
             print("⚠️ WARNING: 'price' column not found. Using dummy target for demo.")
             train_df['price'] = np.random.uniform(10, 500, len(train_df))

    except FileNotFoundError as e:
        print(f"❌ Error: Required data file not found: {e}")
        return

    # --- 2. Target Preprocessing ---
    # We predict the log of (1 + price) for a better distribution
    Y_train_log = np.log1p(train_df['price'])
    
    # --- 3. Feature Engineering ---
    print("\n📝 Extracting Text Features...")
    # NOTE: Pass train_df to fit TFIDF/Brand OHE, then apply to test_df
    X_train_text_sparse, _, tfidf_vectorizer, _ = engineer_text_features(
        train_df, 
        fit_tfidf=True, 
        analyze_importance=False # Set to False for faster training
    )
    X_test_text_sparse, _, _, _ = engineer_text_features(
        test_df, 
        fit_tfidf=False, 
        tfidf_vectorizer=tfidf_vectorizer,
        analyze_importance=False
    ) # Needs to be transformed by same TFIDF

    # --- 4. Image Feature Extraction ---
    print("\n🖼️ Extracting Image Features...")
    # These are dense NumPy arrays
    X_train_image, _ = extract_comprehensive_image_features(train_df)
    X_test_image, _ = extract_comprehensive_image_features(test_df)

    # Explicitly cast image features to float64 to prevent string/object dtype errors
    X_train_image = X_train_image.astype(np.float64)
    X_test_image = X_test_image.astype(np.float64)

    # --- 5. Feature Integration (Combining) ---
    print("\n🔗 Combining Text and Image Features...")
    # Combine sparse text features with dense image features
    X_train = hstack([X_train_text_sparse, X_train_image])
    X_test = hstack([X_test_text_sparse, X_test_image])
    
    print(f"Final Train Feature Shape: {X_train.shape}")
    print(f"Final Test Feature Shape: {X_test.shape}")

    # --- 6. Model Training (LightGBM) ---
    print("\n🧠 Training LightGBM Regressor...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', # Use L1 loss (MAE) which is robust to outliers and similar to SMAPE goal
        metric='mae',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42
    )

    lgbm.fit(X_train, Y_train_log)

    # --- 7. Prediction ---
    print("\n🔮 Generating Predictions...")
    
    # Predict on the log scale
    Y_pred_log = lgbm.predict(X_test)
    
    # Inverse transform: Exponentiate to get back to the original price scale
    Y_pred = np.expm1(Y_pred_log)
    
    # Clip predictions to ensure non-negativity
    Y_pred = np.clip(Y_pred, a_min=0, a_max=None)
    
    # --- 8. Evaluation (on training set for sanity check) ---
    Y_train_pred_log = lgbm.predict(X_train)
    Y_train_pred = np.expm1(Y_train_pred_log)
    
    smape_score = smape(train_df['price'], Y_train_pred)
    print(f"\n✅ Training Set SMAPE Score: {smape_score:.4f}%")
    
    # --- 9. Submission File Generation ---
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': Y_pred
    })
    
    submission_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n📁 Predictions successfully saved to {OUTPUT_PATH}")
    print(f"Sample Output:\n{submission_df.head()}")


if __name__ == "__main__":
    train_and_predict_pipeline()