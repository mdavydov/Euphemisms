#!/usr/bin/env python3
"""
Step 3: Evaluate the quality of the linear regression predictor on test set.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the trained model
print("Loading trained model...")
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load train/test split
print("Loading train/test split...")
with open('train_test_split.pkl', 'rb') as f:
    split_data = pickle.load(f)

X_train = split_data['X_train']
X_test = split_data['X_test']
y_train = split_data['y_train']
y_test = split_data['y_test']

print(f"Test set size: {len(X_test)} examples")

# Make predictions on test set
print("\nEvaluating on test set...")
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# For binary classification task (labels 0/1), calculate accuracy with threshold 0.5
y_pred_binary = (y_pred >= 0.5).astype(int)
accuracy = np.mean(y_pred_binary == y_test)

# Calculate confusion matrix-like statistics
true_positives = np.sum((y_pred_binary == 1) & (y_test == 1))
true_negatives = np.sum((y_pred_binary == 0) & (y_test == 0))
false_positives = np.sum((y_pred_binary == 1) & (y_test == 0))
false_negatives = np.sum((y_pred_binary == 0) & (y_test == 1))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print("\nRegression Metrics:")
print(f"  Mean Squared Error (MSE):  {mse:.4f}")
print(f"  Root Mean Squared Error:   {rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  R² Score:                  {r2:.4f}")

print("\nClassification Metrics (threshold=0.5):")
print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall:     {recall:.4f}")
print(f"  F1 Score:   {f1:.4f}")

print("\nConfusion Matrix:")
print(f"  True Positives:  {true_positives}")
print(f"  True Negatives:  {true_negatives}")
print(f"  False Positives: {false_positives}")
print(f"  False Negatives: {false_negatives}")

# Prediction distribution
print("\nPrediction Statistics:")
print(f"  Min prediction:  {y_pred.min():.4f}")
print(f"  Max prediction:  {y_pred.max():.4f}")
print(f"  Mean prediction: {y_pred.mean():.4f}")
print(f"  Std prediction:  {y_pred.std():.4f}")

print("\nActual label distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# Load original embeddings CSV to get sheet information
print("\nCalculating per-sheet statistics...")
embeddings_df = pd.read_csv('embeddings_with_labels.csv')

# Get embedding columns
embed_cols = [col for col in embeddings_df.columns if col.startswith('emb_')]
X_all = embeddings_df[embed_cols].values
y_all = embeddings_df['label'].values
sheets = embeddings_df['sheet'].values

# Make predictions for all data
y_pred_all = model.predict(X_all)
y_pred_binary_all = (y_pred_all >= 0.5).astype(int)

# Calculate statistics per sheet
sheet_stats = []
for sheet_name in sorted(embeddings_df['sheet'].unique()):
    mask = sheets == sheet_name
    y_true_sheet = y_all[mask]
    y_pred_sheet = y_pred_all[mask]
    y_pred_bin_sheet = y_pred_binary_all[mask]
    
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
sheet_stats_file = 'per_sheet_statistics.csv'
sheet_stats_df.to_csv(sheet_stats_file, index=False)
print(f"Per-sheet statistics saved to {sheet_stats_file}")

# Display summary table
print("\n" + "="*80)
print("PER-SHEET STATISTICS SUMMARY")
print("="*80)
for _, row in sheet_stats_df.iterrows():
    print(f"\n{row['sheet'].upper()}:")
    print(f"  Samples: {row['n_samples']} (Label 0: {row['n_label_0']}, Label 1: {row['n_label_1']})")
    print(f"  Accuracy: {row['accuracy']:.4f} | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | F1: {row['f1_score']:.4f}")
    print(f"  MSE: {row['mse']:.4f} | MAE: {row['mae']:.4f} | R²: {row['r2_score']:.4f}")
    print(f"  Predictions: mean={row['pred_mean']:.4f}, std={row['pred_std']:.4f}, range=[{row['pred_min']:.4f}, {row['pred_max']:.4f}]")
print("="*80)

# Create visualization
print("\nGenerating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test, y_pred, alpha=0.5)
axes[0].plot([0, 1], [0, 1], 'r--', lw=2)
axes[0].set_xlabel('Actual Label')
axes[0].set_ylabel('Predicted Label')
axes[0].set_title('Predicted vs Actual Labels')
axes[0].grid(True, alpha=0.3)

# Plot 2: Prediction distribution
axes[1].hist(y_pred[y_test == 0], alpha=0.5, label='Label 0', bins=20)
axes[1].hist(y_pred[y_test == 1], alpha=0.5, label='Label 1', bins=20)
axes[1].axvline(x=0.5, color='r', linestyle='--', label='Threshold')
axes[1].set_xlabel('Predicted Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Predictions by Actual Label')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150)
print("Visualization saved to model_evaluation.png")

print("\n" + "="*60)
