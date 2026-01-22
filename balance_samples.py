"""Balance and sample labeled data from Excel sheets."""

import pandas as pd
import numpy as np


def balance_and_sample(input_file: str, output_file: str):
    """
    For each sheet in the input Excel file:
    - Count rows with label=0 (N0) and label=1 (N1)
    - Calculate N = min(N0, N1)
    - Randomly select N rows with label=0 and N rows with label=1
    - Save all selected rows to output file
    
    Args:
        input_file: Path to input XLSX file
        output_file: Path to output XLSX file
    """
    # Read the Excel file using the same method as statistics.py
    xls = pd.ExcelFile(input_file)
    
    # Create a writer for the output file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name in xls.sheet_names:
            # Read the sheet
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Check if 'label' column exists
            if 'label' not in df.columns:
                print(f"Warning: Sheet '{sheet_name}' has no 'label' column. Skipping.")
                continue
            
            # Count rows with label=0 and label=1
            df_label_0 = df[df['label'] == 0]
            df_label_1 = df[df['label'] == 1]
            
            N0 = len(df_label_0)
            N1 = len(df_label_1)
            
            print(f"Sheet '{sheet_name}': N0={N0}, N1={N1}")
            
            # Calculate N = min(N0, N1)
            N = min(N0, N1)
            
            if N == 0:
                print(f"  Skipping (N=0)")
                continue
            
            print(f"  Sampling N={N} rows from each label")
            
            # Randomly select N rows from each label
            np.random.seed(42)  # For reproducibility
            selected_0 = df_label_0.sample(n=N, random_state=42)
            selected_1 = df_label_1.sample(n=N, random_state=42)
            
            # Combine selected rows
            selected_df = pd.concat([selected_0, selected_1], ignore_index=True)
            
            # Shuffle the combined dataframe
            selected_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Write to output file
            selected_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Wrote {len(selected_df)} rows to output")
    
    print(f"\nCompleted! Output saved to: {output_file}")


if __name__ == "__main__":
    balance_and_sample("PETs_Ukr.xlsx", "PETs_Ukr_Out.xlsx")
