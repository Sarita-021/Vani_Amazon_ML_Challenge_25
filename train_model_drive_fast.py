import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

# Import feature extraction modules
from src.text_features import engineer_text_features
from src.drive_utils_fast import init_fast_drive_loader, get_images_batch_from_drive

# Import configuration
try:
    from config_local import CREDENTIALS_PATH, FOLDER_ID
except ImportError:
    from config import CREDENTIALS_PATH, FOLDER_ID

# --- Configuration ---
DATASET_FOLDER = 'dataset'
MODELS_FOLDER = 'models'
TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'train.csv')
TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'test.csv')
OUTPUT_PATH = os.path.join(DATASET_FOLDER, 'test_out.csv')

os.makedirs(MODELS_FOLDER, exist_ok=True)

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def extract_basic_image_features_fast(df, batch_size=100):
    """Extract basic image features using fast batch loading"""
    print("🚀 Initializing Fast Drive Loader...")
    init_fast_drive_loader(CREDENTIALS_PATH, FOLDER_ID)
    
    print("📸 Loading images in batches...")
    image_links = df['image_link'].dropna().tolist()
    all_images = get_images_batch_from_drive(image_links, batch_size)
    
    print("🎨 Extracting basic features...")
    features = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        link = row['image_link']
        if pd.notna(link):
            filename = os.path.basename(link)
            img = all_images.get(filename)
            
            if img:
                img_array = np.array(img.convert('RGB').resize((224, 224)))
                feature_row = [
                    np.mean(img_array),  # brightness
                    np.std(img_array),   # contrast
                    img_array.shape[1] / img_array.shape[0],  # aspect ratio
                    np.mean(img_array[:,:,0]),  # red channel
                    np.mean(img_array[:,:,1]),  # green channel
                    np.mean(img_array[:,:,2]),  # blue channel
                ]
            else:
                feature_row = [0, 0, 1.0, 0, 0, 0]
        else:
            feature_row = [0, 0, 1.0, 0, 0, 0]
        
        features.append(feature_row)
    
    return np.array(features)

def train_model():
    """Fast training with basic image features"""
    print("🚀 FAST TRAINING: Loading Training Data...")
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"✓ Loaded {len(train_df)} training samples")
    
    # Target preprocessing
    Y_train_log = np.log1p(train_df['price'])
    
    # Extract text features
    print("\n📝 Extracting Text Features...")
    X_train_text, _, tfidf_vectorizer, feature_columns = engineer_text_features(
        train_df, fit_tfidf=True, analyze_importance=False
    )
    
    # Extract basic image features (faster)
    print("\n🖼️ Extracting Basic Image Features...")
    X_train_image = extract_basic_image_features_fast(train_df)
    
    # Combine features
    X_train = np.hstack([X_train_text, X_train_image])
    print(f"✓ Training features shape: {X_train.shape}")
    
    # Train model
    print("\n🧠 Training LightGBM Model...")
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        metric='mae',
        n_estimators=500,  # Reduced for speed
        learning_rate=0.1,
        num_leaves=31,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, Y_train_log)
    
    # Evaluate
    Y_train_pred_log = model.predict(X_train)
    Y_train_pred = np.expm1(Y_train_pred_log)
    smape_score = smape(train_df['price'], Y_train_pred)
    print(f"✓ Training SMAPE: {smape_score:.4f}%")
    
    # Save components
    print("\n💾 Saving Model...")
    pickle.dump(model, open(os.path.join(MODELS_FOLDER, 'trained_model.pkl'), 'wb'))
    pickle.dump(tfidf_vectorizer, open(os.path.join(MODELS_FOLDER, 'tfidf_vectorizer.pkl'), 'wb'))
    pickle.dump(feature_columns, open(os.path.join(MODELS_FOLDER, 'feature_columns.pkl'), 'wb'))
    
    metadata = {
        'text_features': X_train_text.shape[1],
        'image_features': X_train_image.shape[1],
        'total_features': X_train.shape[1],
        'training_smape': smape_score
    }
    pickle.dump(metadata, open(os.path.join(MODELS_FOLDER, 'model_metadata.pkl'), 'wb'))
    
    print("✅ Fast training completed!")
    return True

def predict_test_data():
    """Fast prediction"""
    print("\n🔮 FAST PREDICTION: Loading Test Data...")
    
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"✓ Loaded {len(test_df)} test samples")
    
    # Load model
    model = pickle.load(open(os.path.join(MODELS_FOLDER, 'trained_model.pkl'), 'rb'))
    tfidf_vectorizer = pickle.load(open(os.path.join(MODELS_FOLDER, 'tfidf_vectorizer.pkl'), 'rb'))
    feature_columns = pickle.load(open(os.path.join(MODELS_FOLDER, 'feature_columns.pkl'), 'rb'))
    metadata = pickle.load(open(os.path.join(MODELS_FOLDER, 'model_metadata.pkl'), 'rb'))
    
    # Extract features
    print("\n📝 Extracting Test Text Features...")
    X_test_text, _, _, _ = engineer_text_features(
        test_df, fit_tfidf=False, tfidf_vectorizer=tfidf_vectorizer, feature_columns=feature_columns
    )
    
    print("\n🖼️ Extracting Test Image Features...")
    X_test_image = extract_basic_image_features_fast(test_df)
    
    # Combine and align
    X_test = np.hstack([X_test_text, X_test_image])
    
    expected_features = metadata['total_features']
    if X_test.shape[1] != expected_features:
        if X_test.shape[1] < expected_features:
            diff = expected_features - X_test.shape[1]
            X_test = np.hstack([X_test, np.zeros((X_test.shape[0], diff))])
        else:
            X_test = X_test[:, :expected_features]
    
    # Predict
    print("\n🎯 Generating Predictions...")
    Y_pred_log = model.predict(X_test)
    Y_pred = np.expm1(Y_pred_log)
    Y_pred = np.clip(Y_pred, a_min=0, a_max=None)
    
    # Save results
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': Y_pred
    })
    submission_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n📁 Predictions saved to {OUTPUT_PATH}")
    print(f"✓ Price range: ${Y_pred.min():.2f} - ${Y_pred.max():.2f}")
    print("✅ Fast prediction completed!")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train_model()
        elif sys.argv[1] == 'predict':
            predict_test_data()
        elif sys.argv[1] == 'full':
            train_model()
            predict_test_data()
    else:
        print("Usage: python train_model_drive_fast.py [train|predict|full]")