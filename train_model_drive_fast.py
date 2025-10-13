import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration
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

def extract_text_features_batch(texts):
    """Process text features in batch for multiprocessing"""
    features = []
    for text in texts:
        text = str(text).lower()
        feature_row = [
            len(text),  # text length
            len(text.split()),  # word count
            text.count('premium') + text.count('luxury'),  # premium indicators
            len(re.findall(r'\d+', text)),  # number count
            1 if any(brand in text for brand in ['apple', 'samsung', 'nike', 'sony']) else 0,  # brand indicator
        ]
        features.append(feature_row)
    return features

def extract_fast_features(df, vectorizer=None, brand_encoder=None, fit=True):
    """Fast feature extraction with multiprocessing"""
    
    # 1. TF-IDF features (reduced size)
    if fit:
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        tfidf_features = vectorizer.fit_transform(df['catalog_content'].fillna(''))
    else:
        tfidf_features = vectorizer.transform(df['catalog_content'].fillna(''))
    
    # 2. Parallel text processing
    texts = df['catalog_content'].fillna('').tolist()
    chunk_size = len(texts) // cpu_count() + 1
    text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    with Pool(processes=min(4, cpu_count())) as pool:  # Limit processes to avoid memory issues
        text_features_chunks = pool.map(extract_text_features_batch, text_chunks)
    
    # Flatten results
    text_features = []
    for chunk in text_features_chunks:
        text_features.extend(chunk)
    text_features = np.array(text_features)
    
    # 3. Simple brand extraction
    brands = []
    for text in df['catalog_content'].fillna(''):
        brand_match = re.search(r'\b([A-Z][a-z]+)\b', text)
        brands.append(brand_match.group(1) if brand_match else 'Unknown')
    
    if fit:
        brand_encoder = LabelEncoder()
        brand_encoded = brand_encoder.fit_transform(brands)
    else:
        brand_encoded = []
        for brand in brands:
            if brand in brand_encoder.classes_:
                brand_encoded.append(brand_encoder.transform([brand])[0])
            else:
                brand_encoded.append(0)
        brand_encoded = np.array(brand_encoded)
    
    # Combine features
    combined_features = np.hstack([
        tfidf_features.toarray(),
        text_features,
        brand_encoded.reshape(-1, 1)
    ])
    
    return combined_features, vectorizer, brand_encoder

def train_model():
    """Fast training with multiprocessing"""
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # Simple outlier removal
    Q1, Q3 = train_df['price'].quantile([0.1, 0.9])
    train_df = train_df[(train_df['price'] >= Q1) & (train_df['price'] <= Q3)]
    print(f"Training samples: {len(train_df)}")
    
    # Extract features
    print("Extracting features with multiprocessing...")
    X, vectorizer, brand_encoder = extract_fast_features(train_df, fit=True)
    print(f"Feature shape: {X.shape}")
    
    # Target transformation - less aggressive log to preserve higher values
    y = np.log(train_df['price'].values + 1)  # Standard log transformation
    
    # Fast model training
    print("Training LightGBM...")
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        n_estimators=500,  # Reduced for speed
        learning_rate=0.1,
        num_leaves=31,
        n_jobs=-1,  # Use all cores
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    
    # Quick evaluation
    pred = np.exp(model.predict(X)) - 1
    pred = np.clip(pred, 3, 100)  # Ensure training predictions are in range
    score = smape(train_df['price'], pred)
    print(f"Training SMAPE: {score:.2f}%")
    
    # Save model
    pickle.dump(model, open(os.path.join(MODELS_FOLDER, 'fast_model.pkl'), 'wb'))
    pickle.dump(vectorizer, open(os.path.join(MODELS_FOLDER, 'fast_vectorizer.pkl'), 'wb'))
    pickle.dump(brand_encoder, open(os.path.join(MODELS_FOLDER, 'fast_brand_encoder.pkl'), 'wb'))
    
    print("Fast model saved!")

def predict():
    """Fast prediction with multiprocessing"""
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Load model
    model = pickle.load(open(os.path.join(MODELS_FOLDER, 'fast_model.pkl'), 'rb'))
    vectorizer = pickle.load(open(os.path.join(MODELS_FOLDER, 'fast_vectorizer.pkl'), 'rb'))
    brand_encoder = pickle.load(open(os.path.join(MODELS_FOLDER, 'fast_brand_encoder.pkl'), 'rb'))
    
    # Extract features
    print("Extracting test features with multiprocessing...")
    X_test, _, _ = extract_fast_features(test_df, vectorizer, brand_encoder, fit=False)
    
    # Predict
    print("Generating predictions...")
    pred = np.exp(model.predict(X_test)) - 1
    
    # Scale predictions to target range (3-100)
    pred_min, pred_max = pred.min(), pred.max()
    pred_scaled = 3 + (pred - pred_min) / (pred_max - pred_min) * 97  # Scale to 3-100 range
    pred = np.clip(pred_scaled, 3, 100)
    
    # Save results
    result_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': pred
    })
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Fast predictions saved! Price range: ${pred.min():.2f} - ${pred.max():.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train_model()
        elif sys.argv[1] == 'predict':
            predict()
        elif sys.argv[1] == 'full':
            train_model()
            predict()
    else:
        print("Usage: python train_model_drive_fast.py [train|predict|full]")