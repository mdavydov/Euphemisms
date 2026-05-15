#!/usr/bin/env python3
"""
Step 2: Train linear regression model on the training embedding data.
Uses embeddings_train.csv (produced from PETs_Ukr_Train.xlsx).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load training embeddings
print("Loading training embeddings from CSV...")
df = pd.read_csv('embeddings_train.csv')
print(f"Loaded {len(df)} training examples")

# Separate features (embeddings) and labels
embedding_cols = [col for col in df.columns if col.startswith('emb_')]
X_train = df[embedding_cols].values
y_train = df['label'].values

print(f"Feature dimension: {X_train.shape[1]}")
print(f"Label distribution:\n{pd.Series(y_train).value_counts()}")

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
