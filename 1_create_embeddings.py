#!/usr/bin/env python3
"""
Step 1: Extract bracketed text from PETs_Ukr.xlsx and create compound embeddings.
Processes all sheets in the Excel file and combines them.
Creates embeddings for both full sentence and bracketed text, then combines them.
Saves compound embeddings and labels to CSV file.
"""

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# Load the Excel file and read all sheets
print("Loading PETs_Ukr.xlsx...")
xl = pd.ExcelFile('PETs_Ukr.xlsx')
print(f"Found {len(xl.sheet_names)} sheets: {xl.sheet_names}")

# Read all sheets and combine them
all_dfs = []
for sheet_name in xl.sheet_names:
    print(f"  Reading sheet: {sheet_name}")
    df_sheet = pd.read_excel(xl, sheet_name)
    # Keep only relevant columns that exist in all sheets
    if 'text' in df_sheet.columns and 'label' in df_sheet.columns:
        df_sheet = df_sheet[['text', 'label']].copy()
        df_sheet['sheet'] = sheet_name  # Track which sheet the data came from
        all_dfs.append(df_sheet)
    else:
        print(f"    Warning: Skipping sheet '{sheet_name}' - missing 'text' or 'label' column")

df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows across all sheets: {len(df)}")

# Extract text in brackets from the 'text' column
def extract_bracketed_text(text):
    """Extract text within angle brackets <...>"""
    if pd.isna(text):
        return None
    match = re.search(r'<(.+?)>', text)
    if match:
        return match.group(1)
    return None

def remove_brackets(text):
    """Remove angle brackets but keep the text inside"""
    if pd.isna(text):
        return None
    return re.sub(r'[<>]', '', text)

print("Extracting bracketed text and full sentences...")
df['bracketed_text'] = df['text'].apply(extract_bracketed_text)
df['full_sentence'] = df['text'].apply(remove_brackets)

# Remove rows where bracketed text couldn't be extracted
df_clean = df.dropna(subset=['bracketed_text', 'full_sentence', 'label']).copy()
print(f"Found {len(df_clean)} valid examples with bracketed text and labels")

# Load multilingual sentence transformer model (supports Ukrainian)
print("Loading multilingual embedding model (LaBSE)...")
model = SentenceTransformer('sentence-transformers/LaBSE')

# Create embeddings for full sentences
print("Creating embeddings for full sentences...")
sentence_embeddings = model.encode(df_clean['full_sentence'].tolist(), show_progress_bar=True)

# Create embeddings for bracketed text
print("Creating embeddings for bracketed text...")
bracket_embeddings = model.encode(df_clean['bracketed_text'].tolist(), show_progress_bar=True)

# Combine embeddings by concatenation
print("Combining embeddings...")
compound_embeddings = np.concatenate([sentence_embeddings, bracket_embeddings], axis=1)
print(f"Compound embedding dimension: {compound_embeddings.shape[1]}")

# Prepare data for CSV
print("Preparing CSV data...")
# Create column names for compound embeddings
embedding_cols = [f'emb_{i}' for i in range(compound_embeddings.shape[1])]

# Create DataFrame with embeddings and labels
embeddings_df = pd.DataFrame(compound_embeddings, columns=embedding_cols)
embeddings_df['full_sentence'] = df_clean['full_sentence'].values
embeddings_df['bracketed_text'] = df_clean['bracketed_text'].values
embeddings_df['label'] = df_clean['label'].values
embeddings_df['sheet'] = df_clean['sheet'].values

# Reorder columns: text and label first, then embeddings
cols = ['sheet', 'full_sentence', 'bracketed_text', 'label'] + embedding_cols
embeddings_df = embeddings_df[cols]

# Save to CSV
output_file = 'embeddings_with_labels.csv'
embeddings_df.to_csv(output_file, index=False)
print(f"\nSaved compound embeddings to {output_file}")
print(f"Shape: {embeddings_df.shape}")
print(f"\nLabel distribution:\n{embeddings_df['label'].value_counts()}")
print(f"\nExamples per sheet:\n{embeddings_df['sheet'].value_counts()}")
