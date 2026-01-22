#!/usr/bin/env python3
"""
Step 2: Train linear regression model using 50% of the embedding data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load embeddings
print("Loading embeddings from CSV...")
df = pd.read_csv('embeddings_with_labels.csv')
print(f"Loaded {len(df)} examples")

# Separate features (embeddings) and labels
embedding_cols = [col for col in df.columns if col.startswith('emb_')]
X = df[embedding_cols].values
y = df['label'].values

print(f"Feature dimension: {X.shape[1]}")
print(f"Label distribution:\n{pd.Series(y).value_counts()}")

# Split data: 50% for training, 50% for testing
print("\nSplitting data (50% train, 50% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
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

# Save train/test split indices for reproducibility
split_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}
split_file = 'train_test_split.pkl'
with open(split_file, 'wb') as f:
    pickle.dump(split_data, f)
print(f"Train/test split saved to {split_file}")

# Quick training set evaluation
train_predictions = model.predict(X_train)
train_mse = np.mean((train_predictions - y_train) ** 2)
train_r2 = model.score(X_train, y_train)

print(f"\nTraining set performance:")
print(f"  MSE: {train_mse:.4f}")
print(f"  R² score: {train_r2:.4f}")
