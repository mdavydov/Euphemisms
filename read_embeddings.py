import pandas as pd
import numpy as np

for csv_file in ['embeddings_train.csv', 'embeddings_test.csv']:
    print(f"\n{'=' * 60}")
    print(f"Reading: {csv_file}")
    print('=' * 60)
    df = pd.read_csv(csv_file)

    # Extract metadata columns
    metadata_cols = ['sheet', 'full_sentence', 'bracketed_text', 'label']
    metadata = df[metadata_cols]

    # Extract embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    embeddings = df[embedding_cols].values

    print(f"Loaded {len(df)} rows")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"\nFirst row metadata:")
    print(metadata.iloc[0])
    print(f"\nFirst embedding (first 10 dims): {embeddings[0][:10]}")
