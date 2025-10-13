import os
import re
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

# Load spaCy model for lightweight NER (ORG, PRODUCT, etc.)
nlp = spacy.load("en_core_web_sm")

# --- Conversion Constants ---
CONVERSION_FACTORS = {
    'kilogram': 1000.0, 'pound': 453.592, 'ounce': 28.3495, 'gram': 1.0, 'milligram': 0.001,
    'liter': 1000.0, 'milliliter': 1.0, 'fluid_ounce': 29.5735,
    'count': 1.0, 'pack': 1.0, 'pair': 1.0, 'other_unit': 1.0, 'missing': 1.0 
}

UNIT_CATEGORIES = {
    'kilogram': 'weight', 'pound': 'weight', 'ounce': 'weight', 'gram': 'weight', 'milligram': 'weight',
    'liter': 'volume', 'milliliter': 'volume', 'fluid_ounce': 'volume',
    'count': 'count', 'pack': 'count', 'pair': 'count',
    'other_unit': 'other', 'missing': 'other'
}

# --- Enhanced Price-indicative Keywords ---
LUXURY_KEYWORDS = ['premium', 'professional', 'deluxe', 'luxury', 'pro', 'advanced', 'elite', 'superior', 'gourmet', 'artisan', 'organic']
BUDGET_KEYWORDS = ['basic', 'economy', 'budget', 'standard', 'simple', 'essential', 'value']
STRONG_BRANDS = ['starbucks', 'coca-cola', 'kellogg', 'nestle', 'pepsi', 'kraft', 'general mills']

def extract_brand(text):
    """Extract probable brand name from product text without predefined list."""
    if not isinstance(text, str):
        return None

    # 1️⃣ Extract 'Item Name' segment if available
    match = re.search(r"Item Name:\s*(.*?)(?:\n|Bullet Point|Product Description|Value|$)", 
                      text, re.IGNORECASE | re.DOTALL)
    item_name = match.group(1).strip() if match else text.strip()

    # 2️⃣ Clean and standardize
    item_name = re.sub(r"[-|–|,|\(|\[].*?$", "", item_name)  # remove trailing junk
    item_name = re.sub(r"\s+", " ", item_name).strip()

    # 3️⃣ Look for brand-like patterns (first capitalized phrase)
    # e.g., "Gift Basket Village", "Bear Creek Country Kitchens"
    pattern = re.match(r"([A-Z][a-zA-Z&\s]{1,40})", item_name)
    if pattern:
        candidate = pattern.group(1).strip()
        # If it’s 1–4 words and not generic, likely a brand
        if len(candidate.split()) <= 4 and not any(w.lower() in ["item", "product", "gift", "set"] for w in candidate.split()):
            return candidate

    # 4️⃣ Fallback: Use spaCy NER
    doc = nlp(item_name)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            return ent.text.strip()

    # 5️⃣ Fallback: first 2–3 words if nothing else
    words = item_name.split()
    return " ".join(words[:3]).strip() if words else None

def extract_category(text):
    """Extracts broad product category from text"""
    if pd.isna(text):
        return "other"

    text = str(text).lower()

    generic_keywords = [
        "electronics", "home", "kitchen", "beauty", "health", "personal care",
        "clothing", "accessories", "sports", "outdoor", "automotive",
        "baby", "grocery", "pet", "office", "furniture", "tools",
        "garden", "toy", "book", "musical", "art", "industrial"
    ]

    for kw in generic_keywords:
        if kw in text:
            return kw

    doc = nlp(text)
    labels = [ent.label_ for ent in doc.ents]
    if "PRODUCT" in labels:
        return "product"
    if "ORG" in labels:
        return "brand_related"

    tokens = re.findall(r"[a-zA-Z]+", text)
    common = Counter(tokens).most_common(10)

    for word, _ in common:
        if any(x in word for x in ["food", "snack", "drink", "chocolate"]):
            return "grocery"
        if any(x in word for x in ["cream", "soap", "shampoo", "makeup"]):
            return "beauty"
        if any(x in word for x in ["shirt", "pant", "dress", "shoe"]):
            return "clothing"
        if any(x in word for x in ["chair", "table", "sofa", "bed", "lamp"]):
            return "home"

    return "other"

def extract_price_indicators(text):
    """Extract price-related signals from text"""
    if pd.isna(text):
        return {'premium_count': 0, 'bulk_indicators': 0, 'size_mentions': 0, 'brand_strength': 0}
    
    text = str(text).lower()
    return {
        'premium_count': len(re.findall(r'premium|gourmet|artisan|organic|luxury', text)),
        'bulk_indicators': len(re.findall(r'pack of \d+|bulk|case of|\d+\s*count', text)),
        'size_mentions': len(re.findall(r'\d+\s*(oz|lb|ml|g|count|fl oz)', text)),
        'brand_strength': 1 if any(brand in text for brand in STRONG_BRANDS) else 0
    }

def cluster_similar_products_optimized(df, text_column="catalog_content", n_clusters=None, fitted_model=None):
    """Optimized semantic clustering with adaptive cluster count"""
    print("🔹 Running optimized semantic clustering...")
    df = df.copy()
    
    # Adaptive cluster count
    if n_clusters is None:
        n_clusters = min(50, max(10, len(df) // 100))
    
    texts = df[text_column].fillna("").astype(str).tolist()
    
    # Use lighter model for speed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=512, show_progress_bar=True)
    
    # Use MiniBatchKMeans for better performance
    if fitted_model is None:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        cluster_labels = kmeans.fit_predict(embeddings)
    else:
        kmeans = fitted_model
        cluster_labels = kmeans.predict(embeddings)
    df["cluster_label"] = cluster_labels
    
    # Generate cluster keywords using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) > 0:
            cluster_tfidf = tfidf[cluster_indices].mean(axis=0)
            top_terms = np.argsort(np.array(cluster_tfidf).ravel())[-5:][::-1]
            cluster_keywords[i] = " ".join(terms[j] for j in top_terms)
        else:
            cluster_keywords[i] = "misc"
    
    df["cluster_keywords"] = df["cluster_label"].map(cluster_keywords)
    
    print(f"   ✓ Created {n_clusters} clusters with {len(embeddings)} products")
    return df, embeddings, cluster_keywords, kmeans

def create_price_range_features(df, cluster_col='cluster_label'):
    """Create price-based cluster statistics for training data"""
    if 'price' in df.columns:
        print("💲 Creating price range features...")
        cluster_stats = df.groupby(cluster_col)['price'].agg(['mean', 'std', 'min', 'max', 'count'])
        df['cluster_price_mean'] = df[cluster_col].map(cluster_stats['mean'])
        df['cluster_price_std'] = df[cluster_col].map(cluster_stats['std']).fillna(0)
        df['cluster_size'] = df[cluster_col].map(cluster_stats['count'])
        print(f"   ✓ Added cluster price statistics")
    return df

def extract_numerical_features(text):
    """Extract numerical specifications"""
    if pd.isna(text):
        return {}
    
    text = str(text)
    features = {}
    
    # Extract dimensions
    dimension_match = re.search(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', text)
    if dimension_match:
        dims = [float(d) for d in dimension_match.groups()]
        features['volume_estimate'] = dims[0] * dims[1] * dims[2]
        features['max_dimension'] = max(dims)
    
    # Extract numbers
    numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
    if numbers:
        features['number_count'] = len(numbers)
        features['max_number'] = max(float(n) for n in numbers)
    
    return features

def calculate_text_richness(text):
    """Calculate text richness metrics"""
    if pd.isna(text):
        return {'char_count': 0, 'word_count': 0, 'sentence_count': 0}
    
    text = str(text)
    return {
        'char_count': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[.!?]+', text))
    }

def count_price_keywords(text):
    """Count luxury and budget keywords"""
    if pd.isna(text):
        return {'luxury_count': 0, 'budget_count': 0}
    
    text = str(text).lower()
    return {
        'luxury_count': sum(1 for word in LUXURY_KEYWORDS if word in text),
        'budget_count': sum(1 for word in BUDGET_KEYWORDS if word in text)
    }

def clean_text_enhanced(text):
    """Enhanced text cleaning"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s\.\-]', ' ', text)
    text = re.sub(r'\b(bullet point \d+|item name|value|unit)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def standardize_unit(unit):
    """Standardize unit abbreviations"""
    if pd.isna(unit) or unit is None:
        return 'missing'
    
    unit = str(unit).lower().strip()
    unit_mapping = {
        'oz': 'ounce', 'ounces': 'ounce', 'fl oz': 'fluid_ounce',
        'lb': 'pound', 'lbs': 'pound', 'ct': 'count', 'counts': 'count',
        'g': 'gram', 'gs': 'gram', 'kg': 'kilogram', 'ml': 'milliliter',
        'mg': 'milligram', 'l': 'liter', 'pk': 'pack', 'packs': 'pack'
    }
    
    return unit_mapping.get(unit, unit if unit in CONVERSION_FACTORS else 'other_unit')

def convert_to_base_value(row):
    """Convert IPQ value to base units"""
    value = row['IPQ_Value']
    standard_unit = row['IPQ_Unit_Standardized']
    conversion_factor = CONVERSION_FACTORS.get(standard_unit, 1.0)
    return value * conversion_factor

def analyze_feature_importance(X, y, feature_names, top_n=20):
    """Quick feature importance analysis"""
    print("🔍 Analyzing feature importance...")
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(rf.feature_importances_)],
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"   ✓ Top {top_n} most important features:")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def engineer_text_features(df: pd.DataFrame, fit_tfidf=True, tfidf_vectorizer=None, 
                                            cluster_stats=None, analyze_importance=False, feature_columns=None):
    """
    Optimized enhanced text feature engineering
    """
    print("🚀 Starting Optimized Enhanced Text Feature Engineering...")
    df = df.copy()
    
    # 1. Extract IPQ features
    print("📊 Extracting IPQ features...")
    regex_value_unit = r"Value: (\d+\.?\d*)\nUnit: ([\w\s]+)\n"
    extracted_features = df['catalog_content'].apply(
        lambda x: re.search(regex_value_unit, str(x)) if pd.notna(x) else None
    )
    
    df['IPQ_Value'] = extracted_features.apply(
        lambda x: float(x.group(1)) if x and x.group(1) else 1.0
    )
    raw_unit = extracted_features.apply(
        lambda x: x.group(2) if x and x.group(2) else None
    )
    
    df['IPQ_Unit_Standardized'] = raw_unit.apply(standardize_unit)
    df['IPQ_Base_Value'] = df.apply(convert_to_base_value, axis=1)
    df['IPQ_Unit_Type'] = df['IPQ_Unit_Standardized'].map(UNIT_CATEGORIES).fillna('other')
    
    print(f"   ✓ IPQ extraction complete. Found {df['IPQ_Value'].notna().sum()} valid IPQ values")
    
    # 2. Extract brand and category
    print("🏷️  Extracting brands and categories...")
    df['brand'] = df['catalog_content'].apply(extract_brand)
    df['category'] = df['catalog_content'].apply(extract_category)
    
    # 3. Optimized semantic clustering
    if fit_tfidf:
        df, embeddings, cluster_keywords, kmeans_model = cluster_similar_products_optimized(df, text_column='catalog_content')
        # Store clustering model
        feature_columns = {'kmeans_model': kmeans_model}
        print(f"   🔍 DEBUG - Training clusters created: {df['cluster_label'].nunique()}")
    else:
        if feature_columns is None or 'kmeans_model' not in feature_columns:
            raise ValueError("feature_columns with kmeans_model must be provided when fit_tfidf=False")
        df, embeddings, cluster_keywords, _ = cluster_similar_products_optimized(
            df, text_column='catalog_content', fitted_model=feature_columns['kmeans_model']
        )
        print(f"   🔍 DEBUG - Test clusters assigned: {df['cluster_label'].nunique()}")
        print(f"   🔍 DEBUG - Test cluster range: {df['cluster_label'].min()} to {df['cluster_label'].max()}")
    
    # 4. Create price range features if training data
    df = create_price_range_features(df)
    
    # 5. Extract price indicators
    print("💰 Extracting price indicators...")
    price_indicators = df['catalog_content'].apply(extract_price_indicators)
    for feature in ['premium_count', 'bulk_indicators', 'size_mentions', 'brand_strength']:
        df[feature] = price_indicators.apply(lambda x: x.get(feature, 0))
    
    # 6. Extract numerical features
    print("🔢 Extracting numerical specifications...")
    numerical_features = df['catalog_content'].apply(extract_numerical_features)
    for feature in ['volume_estimate', 'max_dimension', 'number_count', 'max_number']:
        df[feature] = numerical_features.apply(lambda x: x.get(feature, 0))
    
    # 7. Calculate text richness
    print("📝 Calculating text richness metrics...")
    richness_features = df['catalog_content'].apply(calculate_text_richness)
    for feature in ['char_count', 'word_count', 'sentence_count']:
        df[feature] = richness_features.apply(lambda x: x[feature])
    
    # 8. Count price keywords
    keyword_features = df['catalog_content'].apply(count_price_keywords)
    for feature in ['luxury_count', 'budget_count']:
        df[feature] = keyword_features.apply(lambda x: x[feature])
    
    print(f"   ✓ Extracted {df['premium_count'].gt(0).sum()} products with premium indicators")
    print(f"   ✓ Found {df['bulk_indicators'].gt(0).sum()} products with bulk indicators")
    
    # 9. Optimized TF-IDF
    print("🧹 Enhanced text cleaning and optimized TF-IDF...")
    df['cleaned_content'] = df['catalog_content'].apply(clean_text_enhanced)
    
    if fit_tfidf:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2000,  # Reduced for performance
            stop_words='english',
            min_df=3,
            sublinear_tf=True
        )
        content_tfidf = tfidf.fit_transform(df['cleaned_content'])
        print(f"   ✓ TF-IDF fitted with {content_tfidf.shape[1]} features")
    else:
        if tfidf_vectorizer is None:
            raise ValueError("tfidf_vectorizer must be provided when fit_tfidf=False")
        content_tfidf = tfidf_vectorizer.transform(df['cleaned_content'])
        tfidf = tfidf_vectorizer
        print(f"   ✓ TF-IDF transformed with {content_tfidf.shape[1]} features")
    
    # 10. Create categorical features with consistent columns
    print("🏗️  Creating categorical features...")
    categorical_features = []
    
    if fit_tfidf:
        # Training: create and store feature columns
        unit_type_dummies = pd.get_dummies(df['IPQ_Unit_Type'], prefix='Unit')
        category_dummies = pd.get_dummies(df['category'], prefix='Cat')
        cluster_dummies = pd.get_dummies(df['cluster_label'], prefix='Cluster')
        
        # Top brands only
        top_brands = df['brand'].value_counts().head(5).index
        for brand in top_brands:
            df[f'Brand_{brand}'] = (df['brand'] == brand).astype(int)
        
        # Store feature columns for test consistency
        feature_columns.update({
            'unit_types': unit_type_dummies.columns.tolist(),
            'categories': category_dummies.columns.tolist(), 
            'clusters': cluster_dummies.columns.tolist(),
            'brands': [f'Brand_{brand}' for brand in top_brands]
        })
        
        categorical_features.extend([unit_type_dummies, category_dummies, cluster_dummies])
    else:
        # Test: use stored feature columns for consistency
        if feature_columns is None:
            raise ValueError("feature_columns must be provided when fit_tfidf=False")
        
        # Create dummies with consistent columns
        unit_type_dummies = pd.get_dummies(df['IPQ_Unit_Type'], prefix='Unit')
        for col in feature_columns['unit_types']:
            if col not in unit_type_dummies.columns:
                unit_type_dummies[col] = 0
        unit_type_dummies = unit_type_dummies[feature_columns['unit_types']]
        
        category_dummies = pd.get_dummies(df['category'], prefix='Cat')
        for col in feature_columns['categories']:
            if col not in category_dummies.columns:
                category_dummies[col] = 0
        category_dummies = category_dummies[feature_columns['categories']]
        
        cluster_dummies = pd.get_dummies(df['cluster_label'], prefix='Cluster')
        print(f"   🔍 DEBUG - Test cluster dummies created: {list(cluster_dummies.columns)}")
        print(f"   🔍 DEBUG - Expected cluster columns: {feature_columns['clusters']}")
        for col in feature_columns['clusters']:
            if col not in cluster_dummies.columns:
                cluster_dummies[col] = 0
        cluster_dummies = cluster_dummies[feature_columns['clusters']]
        print(f"   🔍 DEBUG - Final cluster dummies shape: {cluster_dummies.shape}")
        
        # Brand features
        for brand_col in feature_columns['brands']:
            brand = brand_col.replace('Brand_', '')
            df[brand_col] = (df['brand'] == brand).astype(int)
        
        categorical_features.extend([unit_type_dummies, category_dummies, cluster_dummies])
    
    # 11. Combine all features
    print("🔗 Combining all features...")
    numerical_cols = [
        'IPQ_Base_Value', 'volume_estimate', 'max_dimension', 'number_count', 
        'max_number', 'char_count', 'word_count', 'sentence_count',
        'luxury_count', 'budget_count', 'premium_count', 'bulk_indicators', 
        'size_mentions', 'brand_strength'
    ]
    
    # Add cluster price features if available
    if 'cluster_price_mean' in df.columns:
        numerical_cols.extend(['cluster_price_mean', 'cluster_price_std', 'cluster_size'])
    
    # Add brand features
    if fit_tfidf:
        brand_cols = [col for col in df.columns if col.startswith('Brand_')]
    else:
        brand_cols = feature_columns['brands']
    numerical_cols.extend(brand_cols)
    
    numerical_features_array = df[numerical_cols].fillna(0).values.astype(np.float64)
    
    # Handle categorical features safely
    if categorical_features:
        categorical_arrays = [cat_df.values for cat_df in categorical_features]
        categorical_combined = np.hstack(categorical_arrays).astype(np.float64)
        # Memory-optimized feature combination
        dense_features = np.hstack([numerical_features_array, categorical_combined]).astype(np.float64)
    else:
        categorical_combined = np.array([]).reshape(len(df), 0)
        dense_features = numerical_features_array
    final_features_matrix = hstack([content_tfidf, csr_matrix(dense_features)])
    
    # Convert to dense array for LightGBM compatibility
    final_features_matrix = final_features_matrix.toarray()
    
    print(f"✅ Optimized feature engineering complete!")
    print(f"   📏 Final feature matrix shape: {final_features_matrix.shape}")
    print(f"   📊 TF-IDF features: {content_tfidf.shape[1]}")
    print(f"   🔢 Numerical features: {len(numerical_cols)}")
    print(f"   🏷️  Categorical features: {categorical_combined.shape[1] if categorical_features else 0}")
    print(f"   💾 Memory usage: {final_features_matrix.nbytes / 1024 / 1024:.1f} MB")
    
    # Debug feature dimensions
    if not fit_tfidf:
        print(f"   🔍 DEBUG - Dense features shape: {dense_features.shape}")
        print(f"   🔍 DEBUG - Content TF-IDF shape: {content_tfidf.shape}")
        print(f"   🔍 DEBUG - Numerical array shape: {numerical_features_array.shape}")
        if categorical_features:
            print(f"   🔍 DEBUG - Categorical combined shape: {categorical_combined.shape}")
    
    # Feature importance analysis if requested and target available
    feature_names = (['tfidf_' + str(i) for i in range(content_tfidf.shape[1])] + 
                    numerical_cols + 
                    ([f'cat_{i}' for i in range(categorical_combined.shape[1])] if categorical_features else []))
    
    if analyze_importance and 'price' in df.columns:
        importance_df = analyze_feature_importance(final_features_matrix, df['price'], feature_names)
    else:
        importance_df = None
    
    # Create summary dataframe
    summary_cols = numerical_cols + ['IPQ_Unit_Type', 'category', 'brand', 'cluster_label']
    feature_summary = df[[col for col in summary_cols if col in df.columns]].copy()
    
    if fit_tfidf:
        return final_features_matrix, feature_summary, tfidf, feature_columns
    else:
        return final_features_matrix, feature_summary, tfidf, None

if __name__ == "__main__":
    print("🎯 Text Feature Engineering for Production")
    print("=" * 60)
    
    DATASET_FOLDER = 'dataset'
    TRAIN_DATA_PATH = os.path.join(DATASET_FOLDER, 'sample_train.csv')
    TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'sample_test.csv')
    
    # Load training data for fitting transformers
    if os.path.exists(TRAIN_DATA_PATH):
        print(f"📂 Loading training data from: {TRAIN_DATA_PATH}")
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"   ✓ Loaded {len(train_df)} training samples")
        
        # Fit on training data
        train_features, _, tfidf_vectorizer, feature_columns = engineer_text_features(
            train_df, fit_tfidf=True, analyze_importance=False
        )
        print(f"   ✓ Training features shape: {train_features.shape}")
        
        # Transform test data if available
        if os.path.exists(TEST_DATA_PATH):
            print(f"📂 Loading test data from: {TEST_DATA_PATH}")
            test_df = pd.read_csv(TEST_DATA_PATH)
            print(f"   ✓ Loaded {len(test_df)} test samples")
            
            test_features, _, _, _ = engineer_text_features(
                test_df, fit_tfidf=False, tfidf_vectorizer=tfidf_vectorizer, feature_columns=feature_columns
            )
            print(f"   ✓ Test features shape: {test_features.shape}")
    else:
        print(f"❌ Error: Could not find sample_train.csv in 'dataset/' folder")
        exit()
        
        # Run optimized feature engineering
        final_features, feature_summary, tfidf_vectorizer, importance_df = engineer_text_features(
            test_df, analyze_importance=False
        )
        
        print("\n" + "=" * 60)
        print("📋 FEATURE SUMMARY")
        print("=" * 60)
        print(feature_summary.head(10))
        
        print(f"\n🎯 OPTIMIZED FEATURE STATISTICS")
        print(f"   • Total samples: {final_features.shape[0]}")
        print(f"   • Total features: {final_features.shape[1]}")
        print(f"   • Sparse matrix density: {final_features.nnz / (final_features.shape[0] * final_features.shape[1]):.4f}")
        print(f"   • Memory usage: {final_features.data.nbytes / 1024 / 1024:.1f} MB")
        
        # Show insights
        print(f"\n📊 DATA INSIGHTS")
        print(f"   • Products with dimensions: {feature_summary['volume_estimate'].gt(0).sum()}")
        print(f"   • Products with premium indicators: {feature_summary['premium_count'].gt(0).sum()}")
        print(f"   • Products with bulk indicators: {feature_summary['bulk_indicators'].gt(0).sum()}")
        print(f"   • Strong brand products: {feature_summary['brand_strength'].sum()}")
        print(f"   • Average text length: {feature_summary['char_count'].mean():.0f} chars")
        print(f"   • Most common category: {feature_summary['category'].mode().iloc[0]}")
        print(f"   • Number of clusters: {feature_summary['cluster_label'].nunique()}")
        
        print("\n✅ Optimized demo completed successfully!")
        print("🚀 Ready for production use with improved performance and memory efficiency!")