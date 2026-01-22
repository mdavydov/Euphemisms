"""Statistics module for analyzing XLSX file content."""

import pandas as pd
from typing import Dict, Any


def get_file_statistics(input_file: str) -> Dict[str, Any]:
    """
    Analyze the input XLSX file and return comprehensive statistics.
    
    Args:
        input_file: Path to the input XLSX file
        
    Returns:
        Dictionary containing file statistics
    """
    try:
        xls = pd.ExcelFile(input_file)
        
        stats = {
            'total_sheets': len(xls.sheet_names),
            'sheet_names': xls.sheet_names,
            'sheets': {}
        }
        
        total_positive = 0
        total_negative = 0
        total_rows = 0
        
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                sheet_rows = len(df)
                total_rows += sheet_rows
                
                # Count positive and negative examples if 'label' column exists
                positive = negative = 0
                if 'label' in df.columns:
                    positive = int((df['label'] == 1).sum())
                    negative = int((df['label'] == 0).sum())
                    total_positive += positive
                    total_negative += negative
                
                stats['sheets'][sheet_name] = {
                    'rows': sheet_rows,
                    'positive_examples': positive,
                    'negative_examples': negative
                }
                
            except Exception as e:
                stats['sheets'][sheet_name] = {
                    'error': str(e),
                    'rows': 0,
                    'positive_examples': 0,
                    'negative_examples': 0
                }
        
        stats['totals'] = {
            'total_rows': total_rows,
            'total_positive_examples': total_positive,
            'total_negative_examples': total_negative
        }
        
        return stats
        
    except Exception as e:
        return {'error': f'Failed to read file: {str(e)}'}


def print_statistics(stats: Dict[str, Any]) -> None:
    """
    Print formatted statistics to console.
    
    Args:
        stats: Statistics dictionary from get_file_statistics
    """
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        return
    
    print("=== File Statistics ===")
    print(f"Number of sheets: {stats['total_sheets']}")
    print()
    
    print("Sheet details:")
    for sheet_name in stats['sheet_names']:
        sheet_info = stats['sheets'][sheet_name]
        
        if 'error' in sheet_info:
            print(f"  {sheet_name}: ERROR - {sheet_info['error']}")
        else:
            print(f"  {sheet_name}: {sheet_info['rows']} rows")
            if sheet_info['positive_examples'] > 0 or sheet_info['negative_examples'] > 0:
                print(f"    Positive examples: {sheet_info['positive_examples']}")
                print(f"    Negative examples: {sheet_info['negative_examples']}")
    
    print()
    print("=== Totals ===")
    totals = stats['totals']
    print(f"Total rows across all sheets: {totals['total_rows']}")
    if totals['total_positive_examples'] > 0 or totals['total_negative_examples'] > 0:
        print(f"Total positive examples: {totals['total_positive_examples']}")
        print(f"Total negative examples: {totals['total_negative_examples']}")
        print(f"Positive ratio: {totals['total_positive_examples'] / (totals['total_positive_examples'] + totals['total_negative_examples']):.2%}")


if __name__ == "__main__":
    # Test the statistics module
    stats = get_file_statistics("PETs_Ukr.xlsx")
    print_statistics(stats)