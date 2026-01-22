"""Data processing module for Excel files."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
from datetime import datetime
import re

from llm_api import LLMClient, clean_strings, extract_label_from_response
from config import SYSTEM_PROMPT


def calculate_metrics(ground_truth: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """
    Calculate classification metrics from ground truth and predictions.
    
    Args:
        ground_truth: Array of ground truth labels (0 or 1)
        predictions: Array of predicted labels (0 or 1)
        
    Returns:
        Dictionary containing metrics: n, lp, ln, pp, pn, tp, fp, tn, fn, precision, recall, f1
    """
    # Convert to numpy arrays and filter out any None values
    gt = np.array(ground_truth)
    pred = np.array(predictions)
    
    # Find valid indices (where both ground truth and predictions are not None/NaN)
    valid_mask = (~pd.isna(gt)) & (~pd.isna(pred))
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]
    
    n = len(gt_valid)  # total rows with valid data
    
    if n == 0:
        return {
            'n': 0, 'lp': 0, 'ln': 0, 'pp': 0, 'pn': 0,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }
    
    # Ground truth counts
    lp = int(np.sum(gt_valid == 1))  # labeled positives
    ln = int(np.sum(gt_valid == 0))  # labeled negatives
    
    # Prediction counts
    pp = int(np.sum(pred_valid == 1))  # predicted positives
    pn = int(np.sum(pred_valid == 0))  # predicted negatives
    
    # Confusion matrix elements
    tp = int(np.sum((gt_valid == 1) & (pred_valid == 1)))  # true positives
    fp = int(np.sum((gt_valid == 0) & (pred_valid == 1)))  # false positives
    tn = int(np.sum((gt_valid == 0) & (pred_valid == 0)))  # true negatives
    fn = int(np.sum((gt_valid == 1) & (pred_valid == 0)))  # false negatives
    
    # Calculate derived metrics
    precision = tp / pp if pp > 0 else 0.0
    recall = tp / lp if lp > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'n': n, 'lp': lp, 'ln': ln, 'pp': pp, 'pn': pn,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3)
    }


class DataProcessor:
    """Handles Excel file processing with LLM integration."""
    
    def __init__(self, input_file: str, llm_client: Optional[LLMClient] = None, test_mode: bool = False):
        """
        Initialize the data processor.
        
        Args:
            input_file: Path to input Excel file
            llm_client: LLM client for processing (None for statistics-only or test mode)
            test_mode: If True, shows prompts without calling LLM APIs
        """
        self.input_file = input_file
        self.llm_client = llm_client
        self.test_mode = test_mode
        self.processed_rows = 0
        self.total_rows_limit = None
        self.max_rows_per_sheet = None
        self.batch_size = 10  # Default batch size
    
    def set_limits(self, max_total_rows: int, max_rows_per_sheet: int, batch_size: int = 10):
        """Set processing limits."""
        self.total_rows_limit = max_total_rows
        self.max_rows_per_sheet = max_rows_per_sheet
        self.batch_size = batch_size
    
    def process_sheet(self, df: pd.DataFrame, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a single sheet with LLM.
        
        Args:
            df: DataFrame containing the sheet data
            sheet_name: Name of the sheet
            
        Returns:
            Tuple of (DataFrame with added AI columns, metrics dictionary)
        """
        if self.llm_client is None and not self.test_mode:
            raise ValueError("LLM client is required for processing (unless in test mode)")
        
        # Apply row limits
        available_rows = self.total_rows_limit - self.processed_rows if self.total_rows_limit else len(df)
        rows_to_process = min(len(df), self.max_rows_per_sheet or len(df), available_rows)
        
        if rows_to_process <= 0:
            print(f"Skipping {sheet_name}: row limit reached")
            # Return empty metrics for skipped sheets
            metrics = {
                'sheet_name': sheet_name,
                'n': 0, 'lp': 0, 'ln': 0, 'pp': 0, 'pn': 0,
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
            return df, metrics
        
        print(f"Processing {rows_to_process} rows from sheet: {sheet_name}")
        
        # Get the text data to process
        if 'text' not in df.columns:
            raise ValueError(f"Sheet {sheet_name} missing 'text' column")
        
        # Take only the rows we want to process
        process_df = df.iloc[:rows_to_process].copy()
        
        # Clean the text strings
        texts = clean_strings(process_df['text'].tolist())
        labels = np.array(process_df['label'], dtype=np.int8) if 'label' in process_df.columns else None
        
        # Process with LLM in batches or show prompts in test mode
        batch_size = self.batch_size
        ai_labels = []
        ai_replies = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            if self.test_mode:
                # Test mode: show prompts instead of calling API
                for j, text in enumerate(batch_texts):
                    print(f"\n--- Prompt for text {i+j+1} ---")
                    print(f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}")
                    print(f"\nUSER TEXT:\n{text}")
                    print("--- End of Prompt ---\n")
                    
                    # In test mode, we don't make predictions
                    ai_labels.append(0)
                    ai_replies.append("0 Test mode")
            else:
                try:
                    batch_results = self.llm_client.process_batch(batch_texts)
                    
                    for result in batch_results:
                        ai_labels.append(extract_label_from_response(result))
                        ai_replies.append(result)
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Fill with error values for this batch
                    for _ in batch_texts:
                        ai_labels.append(0)
                        ai_replies.append("0 Error")
        
        # Create result dataframe from original
        df_result = df.copy()
        
        # Add new columns if they don't exist
        if 'ai_label' not in df_result.columns:
            df_result['ai_label'] = None
        if 'ai_reply' not in df_result.columns:
            df_result['ai_reply'] = None
            
        # Fill processed rows with results
        for i in range(rows_to_process):
            df_result.iloc[i, df_result.columns.get_loc('ai_label')] = ai_labels[i]
            df_result.iloc[i, df_result.columns.get_loc('ai_reply')] = ai_replies[i]
        
        # Calculate metrics for this sheet if we have labels
        metrics = {'sheet_name': sheet_name}
        if 'label' in df.columns and len(ai_labels) > 0:
            # Only calculate metrics for the rows we actually processed
            ground_truth = df.iloc[:rows_to_process]['label'].values
            predictions = np.array(ai_labels)
            metrics.update(calculate_metrics(ground_truth, predictions))
        else:
            # No ground truth available, just count rows
            metrics.update({
                'n': rows_to_process, 'lp': 0, 'ln': 0, 'pp': sum(ai_labels), 'pn': rows_to_process - sum(ai_labels),
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            })
        
        self.processed_rows += rows_to_process
        return df_result, metrics
    
    def verify_file(self) -> Dict[str, Any]:
        """
        Verify the consistency of the input Excel file.
        
        Checks:
        1. Label column should contain only 0 or 1 values
        2. Text column should contain text with exactly one group in angular brackets <group>
        
        Returns:
            Dictionary containing verification results with details about errors found
        """
        print(f"Verifying file: {self.input_file}")
        
        try:
            # Load Excel file
            xls = pd.ExcelFile(self.input_file)
            verification_results = {
                'valid': True,
                'total_sheets': len(xls.sheet_names),
                'sheets_processed': 0,
                'total_rows': 0,
                'errors': [],
                'warnings': [],
                'sheet_details': []
            }
            
            print(f"Found sheets: {xls.sheet_names}")
            
            for sheet_name in xls.sheet_names:
                print(f"Verifying sheet: {sheet_name}")
                
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    sheet_result = self._verify_sheet(df, sheet_name)
                    verification_results['sheet_details'].append(sheet_result)
                    
                    # Update overall results
                    verification_results['sheets_processed'] += 1
                    verification_results['total_rows'] += len(df)
                    
                    if not sheet_result['valid']:
                        verification_results['valid'] = False
                        verification_results['errors'].extend(sheet_result['errors'])
                    
                    verification_results['warnings'].extend(sheet_result['warnings'])
                    
                except Exception as e:
                    error_msg = f"Error reading sheet {sheet_name}: {e}"
                    verification_results['errors'].append(error_msg)
                    verification_results['valid'] = False
                    print(f"  ERROR: {error_msg}")
            
            return verification_results
            
        except Exception as e:
            return {
                'valid': False,
                'total_sheets': 0,
                'sheets_processed': 0,
                'total_rows': 0,
                'errors': [f"Error opening file: {e}"],
                'warnings': [],
                'sheet_details': []
            }
    
    def _verify_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        Verify a single sheet for data consistency.
        
        Args:
            df: DataFrame containing the sheet data
            sheet_name: Name of the sheet
        
        Returns:
            Dictionary with verification results for this sheet
        """
        result = {
            'sheet_name': sheet_name,
            'valid': True,
            'total_rows': len(df),
            'errors': [],
            'warnings': []
        }
        
        # Check if required columns exist
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            for col in missing_columns:
                error_msg = f"Sheet '{sheet_name}': Missing required column '{col}'"
                result['errors'].append(error_msg)
                print(f"  ERROR: {error_msg}")
            
            # Show available columns for reference
            available_cols = list(df.columns)
            info_msg = f"Sheet '{sheet_name}': Available columns: {available_cols}"
            result['warnings'].append(info_msg)
            print(f"  INFO: {info_msg}")
            
            result['valid'] = False
            return result
        
        print(f"  Found {len(df)} rows in sheet {sheet_name}")
        
        # Verify label column (should contain only 0 or 1)
        label_errors = self._verify_labels(df['label'], sheet_name)
        result['errors'].extend(label_errors)
        if label_errors:
            result['valid'] = False
        
        # Verify text column (should contain exactly one group in angular brackets)
        text_errors, text_warnings = self._verify_text_format(df['text'], sheet_name)
        result['errors'].extend(text_errors)
        result['warnings'].extend(text_warnings)
        if text_errors:
            result['valid'] = False
        
        if result['valid']:
            print(f"  ✓ Sheet {sheet_name} is valid")
        else:
            print(f"  ✗ Sheet {sheet_name} has errors")
        
        return result
    
    def _verify_labels(self, label_series: pd.Series, sheet_name: str) -> List[str]:
        """
        Verify that label column contains only 0 or 1 values.
        
        Args:
            label_series: Pandas series containing label data
            sheet_name: Name of the sheet for error reporting
        
        Returns:
            List of error messages
        """
        errors = []
        
        # Check for null/missing values - report each one individually
        null_indices = label_series.isnull()
        for idx in label_series[null_indices].index:
            excel_row = idx + 2  # Convert to Excel row number (1-indexed + header)
            errors.append(f"Sheet '{sheet_name}', Column 'label', Row {excel_row}: Missing/null value")
        
        # Check for invalid values (not 0 or 1)
        valid_labels = {0, 1, 0.0, 1.0}  # Allow both int and float versions
        
        for idx, value in label_series.items():
            if pd.notnull(value) and value not in valid_labels:
                excel_row = idx + 2  # Convert to Excel row number (1-indexed + header)
                # Try to convert to int to handle float representations
                try:
                    int_value = int(float(value))
                    if int_value not in {0, 1}:
                        errors.append(f"Sheet '{sheet_name}', Column 'label', Row {excel_row}: Invalid value '{value}' (should be 0 or 1)")
                except (ValueError, TypeError):
                    errors.append(f"Sheet '{sheet_name}', Column 'label', Row {excel_row}: Invalid value '{value}' (should be 0 or 1)")
        
        if not errors:
            null_count = label_series.isnull().sum()
            valid_count = len(label_series) - null_count
            zeros = (label_series == 0).sum()
            ones = (label_series == 1).sum()
            print(f"    Labels: {valid_count} valid ({zeros} zeros, {ones} ones)")
        
        return errors
    
    def _verify_text_format(self, text_series: pd.Series, sheet_name: str) -> Tuple[List[str], List[str]]:
        """
        Verify that text column contains exactly one group in angular brackets <group>.
        
        Args:
            text_series: Pandas series containing text data
            sheet_name: Name of the sheet for error reporting
        
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Pattern to match angular brackets content
        angular_bracket_pattern = r'<([^<>]+)>'
        
        valid_texts = 0
        
        # Check for null/missing values - report each one individually
        null_indices = text_series.isnull()
        for idx in text_series[null_indices].index:
            excel_row = idx + 2  # Convert to Excel row number (1-indexed + header)
            errors.append(f"Sheet '{sheet_name}', Column 'text', Row {excel_row}: Missing/null value")
        
        for idx, text in text_series.items():
            if pd.isnull(text):
                continue
                
            excel_row = idx + 2  # Convert to Excel row number (1-indexed + header)
            text_str = str(text).strip()
            
            if not text_str:
                errors.append(f"Sheet '{sheet_name}', Column 'text', Row {excel_row}: Empty text value")
                continue
            
            # Find all angular bracket groups
            matches = re.findall(angular_bracket_pattern, text_str)
            
            if len(matches) == 0:
                # Show truncated text for readability
                preview_text = text_str[:50] + ('...' if len(text_str) > 50 else '')
                errors.append(f"Sheet '{sheet_name}', Column 'text', Row {excel_row}: No angular bracket group found - '{preview_text}'")
            elif len(matches) > 1:
                errors.append(f"Sheet '{sheet_name}', Column 'text', Row {excel_row}: Multiple angular bracket groups found ({len(matches)}): {matches}")
            else:
                # Exactly one match - this is good
                valid_texts += 1
                group_content = matches[0].strip()
                
                # Check for some common issues as warnings
                if not group_content:
                    warnings.append(f"Sheet '{sheet_name}', Column 'text', Row {excel_row}: Empty angular bracket group '<>'")
                elif len(group_content.split()) > 5:  # More than 5 words might be unusual
                    warnings.append(f"Sheet '{sheet_name}', Column 'text', Row {excel_row}: Long angular bracket group ({len(group_content.split())} words) '<{group_content}>'")
        
        if not errors:
            print(f"    Text format: {valid_texts} valid texts with angular bracket groups")
        
        return errors, warnings
    
    def process_file(self) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
        """
        Process the entire Excel file.
        
        Returns:
            Tuple of (Dictionary mapping sheet names to processed DataFrames, List of sheet metrics)
        """
        if self.llm_client is None and not self.test_mode:
            raise ValueError("LLM client is required for processing (unless in test mode)")
        
        # Load Excel file
        xls = pd.ExcelFile(self.input_file)
        processed_sheets = {}
        sheet_metrics = []
        
        print(f"Processing file: {self.input_file}")
        print(f"Found sheets: {xls.sheet_names}")
        print(f"Limits: max_total_rows={self.total_rows_limit}, max_per_sheet={self.max_rows_per_sheet}")
        
        for sheet_name in xls.sheet_names:
            if self.total_rows_limit and self.processed_rows >= self.total_rows_limit:
                print(f"Stopping: reached total row limit ({self.total_rows_limit})")
                break
            
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                processed_df, metrics = self.process_sheet(df, sheet_name)
                processed_sheets[sheet_name] = processed_df
                sheet_metrics.append(metrics)
                
            except Exception as e:
                print(f"Error processing sheet {sheet_name}: {e}")
                # Keep original sheet if processing fails
                processed_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
                # Add empty metrics for failed sheets
                sheet_metrics.append({
                    'sheet_name': sheet_name,
                    'n': 0, 'lp': 0, 'ln': 0, 'pp': 0, 'pn': 0,
                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                    'precision': 0.0, 'recall': 0.0, 'f1': 0.0
                })
        
        print(f"Total rows processed: {self.processed_rows}")
        
        # In test mode, just return empty structures since we're only showing prompts
        if self.test_mode:
            return {}, []
        
        return processed_sheets, sheet_metrics
    
    def save_results(self, processed_sheets: Dict[str, pd.DataFrame], output_file: str, sheet_metrics: List[Dict[str, Any]] = None):
        """
        Save processed results to Excel file and metrics to CSV file.
        
        Args:
            processed_sheets: Dictionary of processed DataFrames
            output_file: Output file path
            sheet_metrics: List of metrics dictionaries for each sheet
        """
        print(f"Saving results to: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in processed_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Results saved successfully!")
        
        # Generate CSV metrics file if metrics are provided
        if sheet_metrics:
            csv_file = output_file.rsplit('.', 1)[0] + '.csv'
            self._save_metrics_csv(sheet_metrics, csv_file)
            print(f"Metrics saved to: {csv_file}")
    
    def _save_metrics_csv(self, sheet_metrics: List[Dict[str, Any]], csv_file: str):
        """
        Save metrics to a semicolon-separated CSV file.
        
        Args:
            sheet_metrics: List of metrics dictionaries for each sheet
            csv_file: Path to output CSV file
        """
        with open(csv_file, 'w', encoding='utf-8') as f:
            # Write header
            header = "sheet_name;n;lp;ln;pp;pn;tp;fp;tn;fn;precision;recall;f1\n"
            f.write(header)
            
            # Accumulate totals
            total_metrics = {
                'n': 0, 'lp': 0, 'ln': 0, 'pp': 0, 'pn': 0,
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
            }
            
            # Write each sheet's metrics
            for metrics in sheet_metrics:
                line = (
                    f"{metrics['sheet_name']};{metrics['n']};{metrics['lp']};{metrics['ln']};"
                    f"{metrics['pp']};{metrics['pn']};{metrics['tp']};{metrics['fp']};"
                    f"{metrics['tn']};{metrics['fn']};{metrics['precision']};{metrics['recall']};{metrics['f1']}\n"
                )
                f.write(line)
                
                # Accumulate for totals
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
            
            # Calculate total precision, recall, F1
            total_precision = total_metrics['tp'] / total_metrics['pp'] if total_metrics['pp'] > 0 else 0.0
            total_recall = total_metrics['tp'] / total_metrics['lp'] if total_metrics['lp'] > 0 else 0.0
            total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
            
            # Write totals row
            total_line = (
                f"TOTAL;{total_metrics['n']};{total_metrics['lp']};{total_metrics['ln']};"
                f"{total_metrics['pp']};{total_metrics['pn']};{total_metrics['tp']};{total_metrics['fp']};"
                f"{total_metrics['tn']};{total_metrics['fn']};{round(total_precision, 3)};{round(total_recall, 3)};{round(total_f1, 3)}\n"
            )
            f.write(total_line)


def generate_output_filename(model_provider: str, model_name: str, method: str, experiment_num: int = 1) -> str:
    """
    Generate output filename according to the required format.
    
    Args:
        model_provider: The model provider (deepseek, openai, gemini)
        model_name: The specific model name
        method: Processing method description
        experiment_num: Experiment number
        
    Returns:
        Formatted filename
    """
    # Create a short model identifier
    model_id = f"{model_provider}-{model_name.replace('/', '-').replace('.', '-')}"
    
    return f"Result-{model_id}-{method}-experiment{experiment_num}.xlsx"


def find_next_experiment_number(model_provider: str, model_name: str, method: str) -> int:
    """
    Find the next available experiment number.
    
    Args:
        model_provider: The model provider
        model_name: The model name
        method: Processing method
        
    Returns:
        Next available experiment number
    """
    experiment_num = 1
    while True:
        filename = generate_output_filename(model_provider, model_name, method, experiment_num)
        if not os.path.exists(filename):
            return experiment_num
        experiment_num += 1


def get_processing_method(max_rows_per_sheet: int, max_total_rows: int) -> str:
    """
    Generate a method description based on processing parameters.
    
    Args:
        max_rows_per_sheet: Maximum rows per sheet
        max_total_rows: Maximum total rows
        
    Returns:
        Method description string
    """
    return f"m{max_rows_per_sheet}-n{max_total_rows}"