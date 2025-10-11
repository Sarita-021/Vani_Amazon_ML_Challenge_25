import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, coo_matrix # Need coo_matrix for internal numpy conversion
import os

# --- Conversion Constants (Standard Physics/Chemistry) ---
CONVERSION_FACTORS = {
    # Weight (Base Unit: Grams)
    'kilogram': 1000.0,
    'pound': 453.592,       
    'ounce': 28.3495,       
    'gram': 1.0,
    'milligram': 0.001,
    
    # Volume (Base Unit: Milliliters)
    'liter': 1000.0,
    'milliliter': 1.0,
    'fluid_ounce': 29.5735, # 1 fl oz = 29.5735 ml
    
    # Count (Base Unit: 1)
    'count': 1.0,
    'pack': 1.0,
    'pair': 1.0,
    'other_unit': 1.0,
    'missing': 1.0 
}

# --- Unit Categories ---
UNIT_CATEGORIES = {
    'kilogram': 'weight', 'pound': 'weight', 'ounce': 'weight', 'gram': 'weight', 'milligram': 'weight',
    'liter': 'volume', 'milliliter': 'volume', 'fluid_ounce': 'volume',
    'count': 'count', 'pack': 'count', 'pair': 'count',
    'other_unit': 'other', 'missing': 'other'
}

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\.,]', '', text) 
    text = re.sub(r'bullet point \d+|item name|value|unit', '', text)
    return text

def standardize_unit(unit):
    """
    Converts various unit abbreviations to a single, standard lowercase unit.
    (FIXED: Added base units to the mapping.)
    """
    if pd.isna(unit) or unit is None:
        return 'missing'
    
    unit = str(unit).lower().strip()
    
    unit_mapping = {
        # Abbreviations and plurals
        'oz': 'ounce', 'ounces': 'ounce',
        'fl oz': 'fluid_ounce', 'fl ounce': 'fluid_ounce',
        'lb': 'pound', 'lbs': 'pound', 
        'ct': 'count', 'counts': 'count',
        'g': 'gram', 'gs': 'gram',
        'kg': 'kilogram', 
        'ml': 'milliliter',
        'mg': 'milligram',
        'l': 'liter',
        'pk': 'pack', 'packs': 'pack',
        'pair': 'pair',
        'ounce': 'ounce', 
        'pound': 'pound', 
        'count': 'count', 
        'gram': 'gram',
        'kilogram': 'kilogram',
        'milliliter': 'milliliter',
        'milligram': 'milligram',
        'liter': 'liter',
        'pack': 'pack',
        'pair': 'pair',
        'fluid_ounce': 'fluid_ounce', # Ensure this base unit is also included
    }
    
    # Return the standardized unit, or 'other_unit' if not found
    return unit_mapping.get(unit, 'other_unit')


def convert_to_base_value(row):
    """
    Converts the IPQ_Value to a base unit value (Grams, Milliliters, or 1 for Count).
    """
    value = row['IPQ_Value']
    standard_unit = row['IPQ_Unit_Standardized']
    
    conversion_factor = CONVERSION_FACTORS.get(standard_unit, 1.0)
    
    return value * conversion_factor
    
def engineer_text_features(df: pd.DataFrame):
    """
    Extracts standardized IPQ/Unit features and unstructured TF-IDF features.
    (This function no longer handles the raw unit extraction, but assumes it has 
     access to the original DataFrame and performs the standardization/conversion.)
    """
    # --- Raw Extraction (Required for the function logic) ---
    regex_value_unit = r"Value: (\d+\.?\d*)\nUnit: ([\w\s]+)\n"
    extracted_features = df['catalog_content'].apply(lambda x: re.search(regex_value_unit, str(x)) if pd.notna(x) else None)
    
    df['IPQ_Value'] = extracted_features.apply(lambda x: float(x.group(1)) if x and x.group(1) else np.nan)
    raw_unit = extracted_features.apply(lambda x: x.group(2) if x and x.group(2) else None)
    
    df['IPQ_Value'] = df['IPQ_Value'].fillna(1.0)
    
    # --- 2. Standardization and Categorization ---
    df['IPQ_Unit_Standardized'] = raw_unit.apply(standardize_unit)
    df['IPQ_Base_Value'] = df.apply(convert_to_base_value, axis=1)
    df['IPQ_Unit_Type'] = df['IPQ_Unit_Standardized'].map(UNIT_CATEGORIES).fillna('other')

    # --- 3. Unstructured Feature Preparation (Text Cleaning) ---
    df['cleaned_content'] = df['catalog_content'].apply(clean_text)

    # --- 4. TF-IDF Vectorization ---
    # NOTE: In a real train/test split, you must fit TFIDF only on the training data 
    # and then transform both train and test.
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),        
        max_features=5000,         
        stop_words='english',
        min_df=5                   
    )
    content_tfidf = tfidf.fit_transform(df['cleaned_content'])
    
    # --- 5. Combining Features ---
    # One-Hot Encode the Unit Type (Weight, Volume, Count, Other)
    unit_type_dummies = pd.get_dummies(df['IPQ_Unit_Type'], prefix='Unit_Type')
    
    # Use the new numerical Base Value feature
    ipq_base_value_array = df[['IPQ_Base_Value']].values
    
    # FIX: Use numpy.hstack (np.hstack) to combine the DENSE features (Dummies and Value).
    # This prevents the scipy.sparse.hstack internal error when called on two dense inputs.
    dense_structured_features = np.hstack([unit_type_dummies.values, ipq_base_value_array])
    
    # Now, combine the sparse TF-IDF matrix with the dense structured features 
    # using scipy.sparse.hstack (aliased as hstack). This function can handle 
    # converting the dense array to sparse for the final assembly.
    final_features_matrix = hstack([content_tfidf, dense_structured_features])
    
    ipq_unit_df = pd.concat([df['IPQ_Base_Value'], df['IPQ_Unit_Type'], unit_type_dummies], axis=1)

    print("Text Feature Engineering complete.")
    return final_features_matrix, ipq_unit_df


if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    TEST_DATA_PATH = os.path.join(DATASET_FOLDER, 'sample_test.csv') 
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Could not find '{TEST_DATA_PATH}'. Please ensure files are in 'dataset/'.")
    else:
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        # --- EDA STEP: Extract and print unique units ---
        print("--- EDA: Unique Raw Units for Mapping Refinement ---")
        regex_value_unit = r"Value: (\d+\.?\d*)\nUnit: ([\w\s]+)\n"
        extracted_features = test_df['catalog_content'].apply(lambda x: re.search(regex_value_unit, str(x)) if pd.notna(x) else None)
        
        # Get the raw unit and convert to lowercase for easy inspection
        raw_units_series = extracted_features.apply(lambda x: x.group(2).lower() if x and x.group(2) else 'missing')
        
        # Print all unique values
        unique_raw_units = sorted(raw_units_series.unique())
        print(f"Total unique raw units found: {len(unique_raw_units)}")
        print(unique_raw_units) 
        print("---------------------------------------------------\n")

        # --- Proceed with Feature Engineering ---
        final_text_features, ipq_structured_features = engineer_text_features(test_df)
        
        print("\n--- Final Text Features Matrix Shape ---")
        print(final_text_features.shape) 
        
        print("\n--- Sample of Advanced Structured IPQ Features ---")
        print(ipq_structured_features.head())