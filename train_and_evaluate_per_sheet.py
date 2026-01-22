#!/usr/bin/env python3
"""
Train linear regression model on 50% of data and evaluate it on words from different sheets.
This script combines training and per-sheet evaluation in one workflow.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

def main():
    # Load embeddings
    print("Loading embeddings from CSV...")
    df = pd.read_csv('embeddings_with_labels.csv')
    print(f"Loaded {len(df)} examples from {df['sheet'].nunique()} sheets")
    
    # Separate features (embeddings) and labels
    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    X = df[embedding_cols].values
    y = df['label'].values
    sheets = df['sheet'].values
    
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Label distribution:\n{pd.Series(y).value_counts()}")
    
    # Split data: 50% for training, 50% for testing
    print("\nSplitting data (50% train, 50% test)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.5, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} examples")
    print(f"Test set: {len(X_test)} examples")
    
    # Train linear regression model
    print("\nTraining linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_file = 'linear_regression_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}")
    
    # Quick training set evaluation
    train_predictions = model.predict(X_train)
    train_mse = np.mean((train_predictions - y_train) ** 2)
    train_r2 = model.score(X_train, y_train)
    
    print(f"\nTraining set performance:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  R² score: {train_r2:.4f}")
    
    # Make predictions only on test set (data not in training)
    print("\nMaking predictions on test set for per-sheet evaluation...")
    y_pred_test = model.predict(X_test)
    y_pred_binary_test = (y_pred_test >= 0.5).astype(int)
    
    # Get sheet information for test set only
    sheets_test = sheets[idx_test]
    
    # Calculate statistics per sheet (only for test set)
    print("\nCalculating per-sheet statistics (test set only)...")
    sheet_stats = []
    
    for sheet_name in sorted(df['sheet'].unique()):
        mask = sheets_test == sheet_name
        
        # Skip if no test samples for this sheet
        if mask.sum() == 0:
            continue
            
        y_true_sheet = y_test[mask]
        y_pred_sheet = y_pred_test[mask]
        y_pred_bin_sheet = y_pred_binary_test[mask]
        
        n_samples = mask.sum()
        n_label_0 = (y_true_sheet == 0).sum()
        n_label_1 = (y_true_sheet == 1).sum()
        
        # Regression metrics
        mse_sheet = mean_squared_error(y_true_sheet, y_pred_sheet)
        mae_sheet = mean_absolute_error(y_true_sheet, y_pred_sheet)
        r2_sheet = r2_score(y_true_sheet, y_pred_sheet)
        
        # Classification metrics
        accuracy_sheet = np.mean(y_pred_bin_sheet == y_true_sheet)
        
        tp = np.sum((y_pred_bin_sheet == 1) & (y_true_sheet == 1))
        tn = np.sum((y_pred_bin_sheet == 0) & (y_true_sheet == 0))
        fp = np.sum((y_pred_bin_sheet == 1) & (y_true_sheet == 0))
        fn = np.sum((y_pred_bin_sheet == 0) & (y_true_sheet == 1))
        
        precision_sheet = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_sheet = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_sheet = 2 * (precision_sheet * recall_sheet) / (precision_sheet + recall_sheet) if (precision_sheet + recall_sheet) > 0 else 0
        
        # Prediction statistics
        pred_mean = y_pred_sheet.mean()
        pred_std = y_pred_sheet.std()
        pred_min = y_pred_sheet.min()
        pred_max = y_pred_sheet.max()
        
        sheet_stats.append({
            'sheet': sheet_name,
            'n_samples': n_samples,
            'n_label_0': n_label_0,
            'n_label_1': n_label_1,
            'mse': mse_sheet,
            'rmse': np.sqrt(mse_sheet),
            'mae': mae_sheet,
            'r2_score': r2_sheet,
            'accuracy': accuracy_sheet,
            'precision': precision_sheet,
            'recall': recall_sheet,
            'f1_score': f1_sheet,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'pred_min': pred_min,
            'pred_max': pred_max
        })
    
    # Create DataFrame and save to CSV
    sheet_stats_df = pd.DataFrame(sheet_stats)
    
    # Calculate total statistics across all test samples
    total_samples = y_test.shape[0]
    total_label_0 = (y_test == 0).sum()
    total_label_1 = (y_test == 1).sum()
    
    total_mse = mean_squared_error(y_test, y_pred_test)
    total_mae = mean_absolute_error(y_test, y_pred_test)
    total_r2 = r2_score(y_test, y_pred_test)
    total_accuracy = np.mean(y_pred_binary_test == y_test)
    
    total_tp = np.sum((y_pred_binary_test == 1) & (y_test == 1))
    total_tn = np.sum((y_pred_binary_test == 0) & (y_test == 0))
    total_fp = np.sum((y_pred_binary_test == 1) & (y_test == 0))
    total_fn = np.sum((y_pred_binary_test == 0) & (y_test == 1))
    
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
    
    total_pred_mean = y_pred_test.mean()
    total_pred_std = y_pred_test.std()
    total_pred_min = y_pred_test.min()
    total_pred_max = y_pred_test.max()
    
    # Add total statistics as the last row
    total_row = pd.DataFrame([{
        'sheet': 'TOTAL',
        'n_samples': total_samples,
        'n_label_0': total_label_0,
        'n_label_1': total_label_1,
        'mse': total_mse,
        'rmse': np.sqrt(total_mse),
        'mae': total_mae,
        'r2_score': total_r2,
        'accuracy': total_accuracy,
        'precision': total_precision,
        'recall': total_recall,
        'f1_score': total_f1,
        'true_positives': total_tp,
        'true_negatives': total_tn,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'pred_mean': total_pred_mean,
        'pred_std': total_pred_std,
        'pred_min': total_pred_min,
        'pred_max': total_pred_max
    }])
    
    sheet_stats_df = pd.concat([sheet_stats_df, total_row], ignore_index=True)
    
    sheet_stats_file = 'per_sheet_statistics_new.csv'
    sheet_stats_df.to_csv(sheet_stats_file, index=False)
    print(f"\nPer-sheet statistics saved to {sheet_stats_file}")
    
    # Display summary table
    print("\n" + "="*80)
    print("PER-SHEET STATISTICS SUMMARY")
    print("="*80)
    for _, row in sheet_stats_df.iterrows():
        print(f"\n{row['sheet'].upper()}:")
        print(f"  Samples: {row['n_samples']} (Label 0: {row['n_label_0']}, Label 1: {row['n_label_1']})")
        print(f"  Accuracy: {row['accuracy']:.4f} ({row['accuracy']*100:.1f}%) | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | F1: {row['f1_score']:.4f}")
        print(f"  MSE: {row['mse']:.4f} | MAE: {row['mae']:.4f} | R²: {row['r2_score']:.4f}")
        print(f"  Predictions: mean={row['pred_mean']:.4f}, std={row['pred_std']:.4f}, range=[{row['pred_min']:.4f}, {row['pred_max']:.4f}]")
    
    print("\n" + "="*80)
    print("\nOVERALL SUMMARY:")
    print(f"  Average Accuracy: {sheet_stats_df['accuracy'].mean():.4f}")
    print(f"  Average F1 Score: {sheet_stats_df['f1_score'].mean():.4f}")
    print(f"  Average R² Score: {sheet_stats_df['r2_score'].mean():.4f}")
    print("="*80)
    
if __name__ == '__main__':
    main()
