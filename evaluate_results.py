#!/usr/bin/env python3
"""
SMAPE Evaluation Script
======================

Evaluates model predictions against correct test labels using SMAPE metric.
Requires test_out_correct.csv file with actual prices for test data.

Usage:
    python evaluate_results.py
"""

import os
import pandas as pd
import numpy as np

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def evaluate_predictions():
    """Evaluate predictions against correct test labels"""
    
    # File paths
    predictions_file = 'dataset/test_out.csv'
    correct_file = 'dataset/test_out_correct.csv'
    
    # Check if files exist
    if not os.path.exists(predictions_file):
        print(f"❌ Error: Predictions file not found: {predictions_file}")
        print("   Please run model training and prediction first.")
        return
    
    if not os.path.exists(correct_file):
        print(f"❌ Error: Correct labels file not found: {correct_file}")
        print("   Please ensure test_out_correct.csv is in the dataset folder.")
        return
    
    # Load data
    try:
        predictions_df = pd.read_csv(predictions_file)
        correct_df = pd.read_csv(correct_file)
        
        print(f"📊 Loaded {len(predictions_df)} predictions")
        print(f"📊 Loaded {len(correct_df)} correct labels")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # Validate data
    if len(predictions_df) != len(correct_df):
        print(f"❌ Error: Mismatch in number of samples")
        print(f"   Predictions: {len(predictions_df)}, Correct: {len(correct_df)}")
        return
    
    # Merge on sample_id
    merged_df = predictions_df.merge(correct_df, on='sample_id', suffixes=('_pred', '_true'))
    
    if len(merged_df) != len(predictions_df):
        print(f"❌ Error: Sample ID mismatch between files")
        return
    
    # Calculate SMAPE
    y_true = merged_df['price_true']
    y_pred = merged_df['price_pred']
    
    smape_score = smape(y_true, y_pred)
    
    # Display results
    print("\n" + "="*50)
    print("📈 SMAPE EVALUATION RESULTS")
    print("="*50)
    print(f"🎯 SMAPE Score: {smape_score:.4f}%")
    print(f"📊 Total Samples: {len(merged_df)}")
    print(f"💰 Price Range (True): ${y_true.min():.2f} - ${y_true.max():.2f}")
    print(f"💰 Price Range (Pred): ${y_pred.min():.2f} - ${y_pred.max():.2f}")
    print(f"📉 Mean Absolute Error: ${np.mean(np.abs(y_true - y_pred)):.2f}")
    print(f"📊 Mean True Price: ${y_true.mean():.2f}")
    print(f"📊 Mean Predicted Price: ${y_pred.mean():.2f}")
    
    # Performance interpretation
    print(f"\n🏆 PERFORMANCE INTERPRETATION:")
    if smape_score < 10:
        print("   🌟 Excellent performance!")
    elif smape_score < 20:
        print("   ✅ Good performance!")
    elif smape_score < 30:
        print("   ⚠️  Fair performance - room for improvement")
    else:
        print("   ❌ Poor performance - significant improvement needed")
    
    # Show sample comparisons
    print(f"\n📋 SAMPLE COMPARISONS (First 10):")
    print("-" * 60)
    print(f"{'Sample ID':<12} {'True Price':<12} {'Pred Price':<12} {'Error %':<10}")
    print("-" * 60)
    
    for i in range(min(10, len(merged_df))):
        sample_id = merged_df.iloc[i]['sample_id']
        true_price = merged_df.iloc[i]['price_true']
        pred_price = merged_df.iloc[i]['price_pred']
        error_pct = abs(true_price - pred_price) / ((abs(true_price) + abs(pred_price)) / 2) * 100
        
        print(f"{sample_id:<12} ${true_price:<11.2f} ${pred_price:<11.2f} {error_pct:<9.1f}%")
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    evaluate_predictions()