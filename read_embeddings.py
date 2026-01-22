import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('embeddings_with_labels.csv')

# Extract metadata columns
metadata_cols = ['sheet', 'full_sentence', 'bracketed_text', 'label']
metadata = df[metadata_cols]

# Extract embedding columns (emb_0 to emb_1535)
embedding_cols = [f'emb_{i}' for i in range(1536)]
embeddings = df[embedding_cols].values

print(f"Loaded {len(df)} rows")
print(f"Metadata shape: {metadata.shape}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"\nFirst row metadata:")
print(metadata.iloc[0])
print(f"\nFirst embedding (first 10 dims): {embeddings[0][:10]}")
