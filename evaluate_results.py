import pandas as pd
import numpy as np

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

# Load the files
predicted = pd.read_csv('dataset/sample_test_out.csv')
correct = pd.read_csv('dataset/sample_test_correct.csv')

# Merge on sample_id to ensure alignment
merged = predicted.merge(correct, on='sample_id', suffixes=('_pred', '_true'))

# Calculate SMAPE
test_smape = smape(merged['price_true'], merged['price_pred'])

print(f"📊 Test Results Analysis")
print(f"=" * 40)
print(f"Total samples: {len(merged)}")
print(f"Test SMAPE: {test_smape:.4f}%")
print(f"")
print(f"Price Statistics:")
print(f"  Predicted - Min: ${merged['price_pred'].min():.2f}, Max: ${merged['price_pred'].max():.2f}, Mean: ${merged['price_pred'].mean():.2f}")
print(f"  Actual    - Min: ${merged['price_true'].min():.2f}, Max: ${merged['price_true'].max():.2f}, Mean: ${merged['price_true'].mean():.2f}")
print(f"")
print(f"Sample comparisons:")
print(merged[['sample_id', 'price_pred', 'price_true']].head(10))