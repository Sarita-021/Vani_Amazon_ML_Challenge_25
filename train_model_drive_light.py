import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import re
import warnings
warnings.filterwarnings('ignore')

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def extract_advanced_features(df, encoders=None, fit=True):
    """Advanced feature extraction with multiple techniques"""
    
    if fit:
        encoders = {}
    
    # 1. Multiple TF-IDF vectorizers
    tfidf_features = []
    
    # Word-level TF-IDF
    if fit:
        encoders['tfidf_word'] = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
        word_tfidf = encoders['tfidf_word'].fit_transform(df['catalog_content'].fillna(''))
    else:
        word_tfidf = encoders['tfidf_word'].transform(df['catalog_content'].fillna(''))
    
    # Character-level TF-IDF
    if fit:
        encoders['tfidf_char'] = TfidfVectorizer(max_features=1000, analyzer='char', ngram_range=(3,5))
        char_tfidf = encoders['tfidf_char'].fit_transform(df['catalog_content'].fillna(''))
    else:
        char_tfidf = encoders['tfidf_char'].transform(df['catalog_content'].fillna(''))
    
    # 2. Advanced text features
    text_features = []
    for text in df['catalog_content'].fillna(''):
        features = [
            len(text),  # text length
            len(text.split()),  # word count
            len(set(text.split())),  # unique words
            text.count('.'),  # sentences
            text.count(','),  # commas
            len(re.findall(r'\d+', text)),  # numbers count
            len(re.findall(r'[A-Z]', text)),  # uppercase letters
            text.count('!') + text.count('?'),  # exclamation/question
        ]
        text_features.append(features)
    
    # 3. Brand extraction (enhanced)
    brands = []
    brand_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Title case
        r'Brand:\s*([A-Za-z\s]+)',  # Brand: label
        r'by\s+([A-Z][a-z]+)',  # by Brand
    ]
    
    for text in df['catalog_content'].fillna(''):
        found_brand = 'Unknown'
        for pattern in brand_patterns:
            match = re.search(pattern, text)
            if match:
                found_brand = match.group(1).strip()
                break
        brands.append(found_brand)
    
    if fit:
        encoders['brand'] = LabelEncoder()
        brand_encoded = encoders['brand'].fit_transform(brands)
    else:
        brand_encoded = []
        for brand in brands:
            if brand in encoders['brand'].classes_:
                brand_encoded.append(encoders['brand'].transform([brand])[0])
            else:
                brand_encoded.append(0)
        brand_encoded = np.array(brand_encoded)
    
    # 4. Price indicators
    price_indicators = []
    price_words = {
        'premium': 3, 'luxury': 4, 'deluxe': 3, 'professional': 2,
        'pro': 2, 'advanced': 2, 'basic': -1, 'standard': 0,
        'cheap': -2, 'budget': -1, 'economy': -1, 'value': -1
    }
    
    for text in df['catalog_content'].fillna(''):
        text_lower = text.lower()
        score = sum(price_words.get(word, 0) for word in text_lower.split())
        price_indicators.append(score)
    
    # 5. Quantity and measurements
    quantities = []
    measurements = []
    
    for text in df['catalog_content'].fillna(''):
        # Quantity
        qty_patterns = [
            r'(\d+)\s*(?:pack|pcs|pieces|count|ct|box)',
            r'pack\s*of\s*(\d+)',
            r'(\d+)\s*in\s*1'
        ]
        qty = 1
        for pattern in qty_patterns:
            match = re.search(pattern, text.lower())
            if match:
                qty = int(match.group(1))
                break
        quantities.append(qty)
        
        # Measurements (size indicators)
        size_score = 0
        size_words = ['large', 'xl', 'big', 'jumbo', 'giant', 'mega']
        small_words = ['small', 'mini', 'tiny', 'compact']
        
        text_lower = text.lower()
        size_score += sum(2 for word in size_words if word in text_lower)
        size_score -= sum(1 for word in small_words if word in text_lower)
        measurements.append(size_score)
    
    # 6. Category classification (enhanced)
    categories = []
    detailed_categories = {
        'electronics': ['electronic', 'digital', 'tech', 'device', 'gadget', 'computer', 'phone'],
        'clothing': ['shirt', 'dress', 'clothing', 'apparel', 'fashion', 'wear', 'fabric'],
        'home_kitchen': ['home', 'kitchen', 'furniture', 'decor', 'appliance'],
        'beauty_health': ['beauty', 'cosmetic', 'skincare', 'health', 'wellness'],
        'sports_outdoor': ['sport', 'fitness', 'exercise', 'outdoor', 'athletic'],
        'books_media': ['book', 'dvd', 'cd', 'media', 'magazine'],
        'toys_games': ['toy', 'game', 'play', 'puzzle', 'doll'],
        'automotive': ['car', 'auto', 'vehicle', 'motor', 'tire']
    }
    
    for text in df['catalog_content'].fillna(''):
        text_lower = text.lower()
        found_category = 'other'
        max_matches = 0
        for cat, keywords in detailed_categories.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > max_matches:
                max_matches = matches
                found_category = cat
        categories.append(found_category)
    
    if fit:
        encoders['category'] = LabelEncoder()
        cat_encoded = encoders['category'].fit_transform(categories)
    else:
        cat_encoded = []
        for cat in categories:
            if cat in encoders['category'].classes_:
                cat_encoded.append(encoders['category'].transform([cat])[0])
            else:
                cat_encoded.append(0)
        cat_encoded = np.array(cat_encoded)
    
    # 7. Text clustering
    if fit:
        # Use SVD for dimensionality reduction before clustering
        encoders['svd'] = TruncatedSVD(n_components=50, random_state=42)
        word_tfidf_reduced = encoders['svd'].fit_transform(word_tfidf)
        
        encoders['kmeans'] = KMeans(n_clusters=20, random_state=42, n_init=10)
        clusters = encoders['kmeans'].fit_predict(word_tfidf_reduced)
    else:
        word_tfidf_reduced = encoders['svd'].transform(word_tfidf)
        clusters = encoders['kmeans'].predict(word_tfidf_reduced)
    
    # 8. Statistical features from text
    stat_features = []
    for text in df['catalog_content'].fillna(''):
        words = text.split()
        if words:
            word_lengths = [len(word) for word in words]
            features = [
                np.mean(word_lengths),  # avg word length
                np.std(word_lengths) if len(word_lengths) > 1 else 0,  # std word length
                max(word_lengths),  # max word length
                min(word_lengths),  # min word length
            ]
        else:
            features = [0, 0, 0, 0]
        stat_features.append(features)
    
    # Combine all features
    text_features = np.array(text_features)
    stat_features = np.array(stat_features)
    
    additional_features = np.column_stack([
        brand_encoded,
        price_indicators,
        quantities,
        measurements,
        cat_encoded,
        clusters
    ])
    
    # Scale numerical features
    if fit:
        encoders['scaler'] = StandardScaler()
        scaled_features = encoders['scaler'].fit_transform(
            np.hstack([text_features, stat_features, additional_features])
        )
    else:
        scaled_features = encoders['scaler'].transform(
            np.hstack([text_features, stat_features, additional_features])
        )
    
    # Combine all features
    all_features = np.hstack([
        word_tfidf.toarray(),
        char_tfidf.toarray(),
        scaled_features
    ])
    
    return all_features, encoders

def train_model():
    """Train ensemble of models"""
    print("Loading training data...")
    train_df = pd.read_csv('dataset/train.csv')
    
    # Advanced outlier removal
    Q1 = train_df['price'].quantile(0.05)
    Q3 = train_df['price'].quantile(0.95)
    train_df = train_df[(train_df['price'] >= Q1) & (train_df['price'] <= Q3)]
    print(f"Training samples after outlier removal: {len(train_df)}")
    
    # Extract features
    print("Extracting advanced features...")
    X, encoders = extract_advanced_features(train_df, fit=True)
    print(f"Feature shape: {X.shape}")
    
    # Target transformation
    y_log = np.log1p(train_df['price'].values)
    
    # Train single model
    print("Training LightGBM...")
    X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=1,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
    )
    
    # Evaluate single model
    pred = np.expm1(lgb_model.predict(X_val))
    y_val_original = np.expm1(y_val)
    score = smape(y_val_original, pred)
    print(f"Validation SMAPE: {score:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    pickle.dump(lgb_model, open('models/advanced_model.pkl', 'wb'))
    pickle.dump(encoders, open('models/advanced_encoders.pkl', 'wb'))
    
    print("Advanced model saved!")

def predict():
    """Advanced single model prediction"""
    print("Loading test data...")
    test_df = pd.read_csv('dataset/test.csv')
    
    # Load model
    model = pickle.load(open('models/advanced_model.pkl', 'rb'))
    encoders = pickle.load(open('models/advanced_encoders.pkl', 'rb'))
    
    # Extract features
    print("Extracting advanced features...")
    X_test, _ = extract_advanced_features(test_df, encoders, fit=False)
    
    # Single model prediction
    print("Generating predictions...")
    pred = np.expm1(model.predict(X_test))
    pred = np.clip(pred, 0.01, None)
    
    # Save results
    result_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': pred
    })
    result_df.to_csv('dataset/test_out.csv', index=False)
    print(f"Advanced predictions saved! Price range: ${pred.min():.2f} - ${pred.max():.2f}")

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
        print("Usage: python train_advanced.py [train|predict|full]")