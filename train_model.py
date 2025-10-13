import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import QuantileTransformer

# Import your feature extraction modules
from src.text_features import engineer_text_features
from src.image_features import extract_comprehensive_image_features

# --- Configuration ---
DATASET_FOLDER = 'dataset'
TRAIN_IMAGES_FOLDER = 'images/train'
TEST_IMAGES_FOLDER = 'images/test'
MODELS_FOLDER = 'models'
TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'train.csv')
TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'test.csv')
OUTPUT_PATH = os.path.join(DATASET_FOLDER, 'test_out.csv')

# Create necessary directories
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(TRAIN_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)

# --- SMAPE Metric Function ---
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    # Clip predictions to prevent division by zero or log(0) issues
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def train_model():
    """Train the model and save all learned components"""
    print("🚀 TRAINING PHASE: Loading Training Data...")
    
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        if 'price' not in train_df.columns:
            print("❌ ERROR: 'price' column not found in training data.")
            return False
        print(f"✓ Loaded {len(train_df)} training samples")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    
    # Target preprocessing
    Y_train_log = np.log1p(train_df['price'])
    
    # Extract and fit text features
    print("\n📝 Training Text Feature Pipeline...")
    X_train_text, _, tfidf_vectorizer, feature_columns = engineer_text_features(
        train_df, fit_tfidf=True, analyze_importance=False
    )
    
    # Extract training image features
    print("\n🖼️ Extracting Training Image Features...")
    X_train_image, feature_names = extract_comprehensive_image_features(
        train_df, use_deep_features=True, model_name='resnet50', images_folder=TRAIN_IMAGES_FOLDER
    )
    X_train_image = X_train_image.astype(np.float64)
    
    # Combine features using numpy concatenation for dense arrays
    X_train = np.hstack([X_train_text, X_train_image])
    print(f"✓ Training features shape: {X_train.shape}")
    
    # Train model
    print("\n🧠 Training LightGBM Model...")
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        metric='mae',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, Y_train_log)
    
    # Evaluate on training set
    Y_train_pred_log = model.predict(X_train)
    Y_train_pred = np.expm1(Y_train_pred_log)
    smape_score = smape(train_df['price'], Y_train_pred)
    print(f"✓ Training SMAPE: {smape_score:.4f}%")
    
    # Save all trained components
    print("\n💾 Saving Trained Model and Components...")
    pickle.dump(model, open(os.path.join(MODELS_FOLDER, 'trained_model.pkl'), 'wb'))
    pickle.dump(tfidf_vectorizer, open(os.path.join(MODELS_FOLDER, 'tfidf_vectorizer.pkl'), 'wb'))
    pickle.dump(feature_columns, open(os.path.join(MODELS_FOLDER, 'feature_columns.pkl'), 'wb'))
    
    # Save training metadata
    metadata = {
        'text_features': X_train_text.shape[1],
        'image_features': X_train_image.shape[1],
        'total_features': X_train.shape[1],
        'training_smape': smape_score
    }
    pickle.dump(metadata, open(os.path.join(MODELS_FOLDER, 'model_metadata.pkl'), 'wb'))
    
    print("✅ Training completed and model saved!")
    return True

def predict_test_data():
    """Load trained model and predict on test data"""
    print("\n🔮 PREDICTION PHASE: Loading Test Data...")
    
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        print(f"✓ Loaded {len(test_df)} test samples")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    
    # Load trained components
    print("\n📦 Loading Trained Model and Components...")
    try:
        model = pickle.load(open(os.path.join(MODELS_FOLDER, 'trained_model.pkl'), 'rb'))
        tfidf_vectorizer = pickle.load(open(os.path.join(MODELS_FOLDER, 'tfidf_vectorizer.pkl'), 'rb'))
        feature_columns = pickle.load(open(os.path.join(MODELS_FOLDER, 'feature_columns.pkl'), 'rb'))
        metadata = pickle.load(open(os.path.join(MODELS_FOLDER, 'model_metadata.pkl'), 'rb'))
        print(f"✓ Loaded model with {metadata['total_features']} features")
    except FileNotFoundError:
        print("❌ ERROR: Trained model not found. Please run training first.")
        return False
    
    # Extract test features using trained transformers
    print("\n📝 Extracting Test Text Features...")
    X_test_text, _, _, _ = engineer_text_features(
        test_df, fit_tfidf=False, tfidf_vectorizer=tfidf_vectorizer, feature_columns=feature_columns
    )
    
    print("\n🖼️ Extracting Test Image Features...")
    X_test_image, _ = extract_comprehensive_image_features(
        test_df, use_deep_features=True, model_name='resnet50', images_folder=TEST_IMAGES_FOLDER
    )
    X_test_image = X_test_image.astype(np.float64)
    
    # Combine test features using numpy concatenation for dense arrays
    X_test = np.hstack([X_test_text, X_test_image])
    print(f"✓ Test features shape: {X_test.shape}")
        
        # --- Align Test Features with Training ---
    print("\n🧩 Aligning Test Features with Trained Model...")

    # Load training metadata
    expected_features = metadata['total_features']

    # Compare shapes
    print(f"Expected features: {expected_features}, Test features: {X_test.shape[1]}")

    # Handle missing features by padding zeros if fewer features exist
    if X_test.shape[1] < expected_features:
        diff = expected_features - X_test.shape[1]
        print(f"⚠️ Test data has {diff} fewer features. Padding with zeros...")
        X_test = np.hstack([X_test, np.zeros((X_test.shape[0], diff))])

    # Handle extra features (should rarely happen)
    elif X_test.shape[1] > expected_features:
        diff = X_test.shape[1] - expected_features
        print(f"⚠️ Test data has {diff} extra features. Trimming to match...")
        X_test = X_test[:, :expected_features]

    print(f"✓ Aligned test features shape: {X_test.shape}")
    
    # Generate predictions
    print("\n🎯 Generating Predictions...")
    Y_pred_log = model.predict(X_test)
    Y_pred = np.expm1(Y_pred_log)
    Y_pred = np.clip(Y_pred, a_min=0, a_max=None)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': Y_pred
    })
    submission_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n📁 Predictions saved to {OUTPUT_PATH}")
    print(f"✓ Total predictions: {len(submission_df)}")
    print(f"✓ Price range: ${Y_pred.min():.2f} - ${Y_pred.max():.2f}")
    print(f"\nSample predictions:\n{submission_df.head()}")
    
    print("\n✅ Prediction completed successfully!")
    return True

def train_and_predict_pipeline():
    """Complete pipeline: train model then predict"""
    print("🚀 Starting Complete ML Pipeline...")
    
    # Phase 1: Training
    if not train_model():
        print("❌ Training failed!")
        return
    
    # Phase 2: Prediction
    if not predict_test_data():
        print("❌ Prediction failed!")
        return
    
    print("\n🎉 Complete pipeline finished successfully!")
    print("🚀 Ready for ML Challenge submission!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train_model()
        elif sys.argv[1] == 'predict':
            predict_test_data()
        elif sys.argv[1] == 'full':
            train_and_predict_pipeline()
        else:
            print("Usage: python train_model.py [train|predict|full]")
            print("  train   - Train model and save components")
            print("  predict - Load trained model and predict on test data")
            print("  full    - Run complete pipeline (train + predict)")
    else:
        print("Please specify mode: python train_model.py [train|predict|full]")