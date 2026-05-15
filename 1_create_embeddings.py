#!/usr/bin/env python3
"""
Step 1: Extract bracketed text from PETs_Ukr_Train.xlsx and PETs_Ukr_Test.xlsx
and create compound embeddings.
Processes all sheets in each Excel file and combines them.
Creates embeddings for both full sentence and bracketed text, then combines them.
Saves compound embeddings and labels to separate CSV files for train and test.
"""

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer


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


def load_and_clean(xlsx_path):
    """Load all sheets from an Excel file and return a cleaned DataFrame."""
    print(f"Loading {xlsx_path}...")
    xl = pd.ExcelFile(xlsx_path)
    print(f"Found {len(xl.sheet_names)} sheets: {xl.sheet_names}")

    all_dfs = []
    for sheet_name in xl.sheet_names:
        print(f"  Reading sheet: {sheet_name}")
        df_sheet = pd.read_excel(xl, sheet_name)
        if 'text' in df_sheet.columns and 'label' in df_sheet.columns:
            df_sheet = df_sheet[['text', 'label']].copy()
            df_sheet['sheet'] = sheet_name
            all_dfs.append(df_sheet)
        else:
            print(f"    Warning: Skipping sheet '{sheet_name}' - missing 'text' or 'label' column")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows across all sheets: {len(df)}")

    df['bracketed_text'] = df['text'].apply(extract_bracketed_text)
    df['full_sentence'] = df['text'].apply(remove_brackets)

    df_clean = df.dropna(subset=['bracketed_text', 'full_sentence', 'label']).copy()
    print(f"Found {len(df_clean)} valid examples with bracketed text and labels")
    return df_clean


def create_embeddings(df_clean, model, output_file):
    """Create compound embeddings and save to CSV."""
    print(f"\nCreating embeddings for {len(df_clean)} examples -> {output_file}")

    sentence_embeddings = model.encode(df_clean['full_sentence'].tolist(), show_progress_bar=True)
    bracket_embeddings = model.encode(df_clean['bracketed_text'].tolist(), show_progress_bar=True)

    compound_embeddings = np.concatenate([sentence_embeddings, bracket_embeddings], axis=1)
    print(f"Compound embedding dimension: {compound_embeddings.shape[1]}")

    embedding_cols = [f'emb_{i}' for i in range(compound_embeddings.shape[1])]
    embeddings_df = pd.DataFrame(compound_embeddings, columns=embedding_cols)
    embeddings_df['full_sentence'] = df_clean['full_sentence'].values
    embeddings_df['bracketed_text'] = df_clean['bracketed_text'].values
    embeddings_df['label'] = df_clean['label'].values
    embeddings_df['sheet'] = df_clean['sheet'].values

    cols = ['sheet', 'full_sentence', 'bracketed_text', 'label'] + embedding_cols
    embeddings_df = embeddings_df[cols]

    embeddings_df.to_csv(output_file, index=False)
    print(f"Saved compound embeddings to {output_file}")
    print(f"Shape: {embeddings_df.shape}")
    print(f"Label distribution:\n{embeddings_df['label'].value_counts()}")
    print(f"Examples per sheet:\n{embeddings_df['sheet'].value_counts()}")


# Load multilingual sentence transformer model (supports Ukrainian)
print("Loading multilingual embedding model (LaBSE)...")
model = SentenceTransformer('sentence-transformers/LaBSE')

# Process training data
df_train = load_and_clean('PETs_Ukr_Train.xlsx')
create_embeddings(df_train, model, 'embeddings_train.csv')

# Process test data
df_test = load_and_clean('PETs_Ukr_Test.xlsx')
create_embeddings(df_test, model, 'embeddings_test.csv')
