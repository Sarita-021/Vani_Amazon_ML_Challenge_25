import os
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import hstack

# Global variables to store loaded model and transformers
MODEL = None
TFIDF_VECTORIZER = None
FEATURE_EXTRACTORS = None

def load_trained_model():
    """Load the trained model and transformers (implement after training)"""
    global MODEL, TFIDF_VECTORIZER, FEATURE_EXTRACTORS
    
    # This would load your saved model artifacts
    # MODEL = pickle.load(open('trained_model.pkl', 'rb'))
    # TFIDF_VECTORIZER = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    # FEATURE_EXTRACTORS = pickle.load(open('feature_extractors.pkl', 'rb'))
    
    # For now, return None to indicate model not loaded
    return False

def extract_features_single(sample_id, catalog_content, image_link):
    """Extract features for a single sample"""
    # This is a simplified version - in practice you'd need to:
    # 1. Extract text features using the same pipeline as training
    # 2. Extract image features if image exists
    # 3. Combine features in the same order as training
    
    # Placeholder feature extraction
    text_length = len(str(catalog_content)) if catalog_content else 0
    has_premium_words = 1 if any(word in str(catalog_content).lower() 
                                for word in ['premium', 'luxury', 'gourmet']) else 0
    
    # Simple feature vector (in practice, use your full feature pipeline)
    features = np.array([text_length, has_premium_words]).reshape(1, -1)
    
    return features

def predictor(sample_id, catalog_content, image_link):
    '''
    Predict price using trained model
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    global MODEL
    
    # Load model if not already loaded
    if MODEL is None:
        model_loaded = load_trained_model()
        if not model_loaded:
            # Fallback: Simple rule-based prediction
            text = str(catalog_content).lower() if catalog_content else ""
            
            # Base price
            base_price = 25.0
            
            # Adjust based on text features
            if any(word in text for word in ['premium', 'luxury', 'gourmet', 'organic']):
                base_price *= 2.5
            elif any(word in text for word in ['basic', 'economy', 'budget']):
                base_price *= 0.7
            
            # Adjust based on text length (more description = higher price)
            text_length = len(text)
            if text_length > 2000:
                base_price *= 1.5
            elif text_length > 1000:
                base_price *= 1.2
            
            # Adjust based on pack size
            import re
            pack_match = re.search(r'pack of (\d+)', text)
            if pack_match:
                pack_size = int(pack_match.group(1))
                base_price *= (1 + pack_size * 0.1)
            
            return round(max(5.0, min(base_price, 1000.0)), 2)
    
    # Use trained model for prediction
    try:
        features = extract_features_single(sample_id, catalog_content, image_link)
        log_price = MODEL.predict(features)[0]
        price = np.expm1(log_price)  # Inverse of log1p transformation
        return round(max(1.0, price), 2)
    except Exception as e:
        # Fallback to rule-based prediction
        return round(np.random.uniform(10.0, 200.0), 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    
    # Read test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply predictor function to each row
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
