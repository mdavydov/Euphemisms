#!/usr/bin/env python3
"""
Main processing script for euphemism analysis.

Usage examples:
    python process.py -model openai -m 10 -n 100 PETs_Ukr.xlsx
    python process.py -model deepseek -m 5 -n 50 PETs_Ukr.xlsx
    python process.py -model gemini -stat PETs_Ukr.xlsx
    python process.py -model mamaylm -m 5 -n 50 PETs_Ukr.xlsx
    python process.py -stat PETs_Ukr.xlsx
    python process.py -verify PETs_Ukr.xlsx
"""

import argparse
import sys
import os
from dotenv import load_dotenv

from config import DEFAULT_MAX_ROWS_PER_SHEET, DEFAULT_MAX_TOTAL_ROWS, SUPPORTED_MODELS
from statistics import get_file_statistics, print_statistics
from llm_api import create_llm_client, get_api_key
from data_processor import DataProcessor, generate_output_filename, find_next_experiment_number, get_processing_method


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process XLSX file with LLM for euphemism analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -model openai -m 10 -n 100 PETs_Ukr.xlsx
  %(prog)s -model deepseek -m 5 -n 50 PETs_Ukr.xlsx
  %(prog)s -model gemini PETs_Ukr.xlsx
  %(prog)s -stat PETs_Ukr.xlsx
  %(prog)s -verify PETs_Ukr.xlsx
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Input XLSX file to process'
    )
    
    parser.add_argument(
        '-m', '--max-rows-per-sheet',
        type=int,
        default=DEFAULT_MAX_ROWS_PER_SHEET,
        help=f'Maximum rows per sheet to process (default: {DEFAULT_MAX_ROWS_PER_SHEET})'
    )
    
    parser.add_argument(
        '-n', '--max-total-rows',
        type=int,
        default=DEFAULT_MAX_TOTAL_ROWS,
        help=f'Maximum total rows to process across all sheets (default: {DEFAULT_MAX_TOTAL_ROWS})'
    )
    
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing (default: 10, lower for less memory usage)'
    )
    
    parser.add_argument(
        '-model', '--model',
        choices=list(SUPPORTED_MODELS.keys()),
        help=f'Model provider to use: {list(SUPPORTED_MODELS.keys())}'
    )
    
    parser.add_argument(
        '-stat', '--statistics-only',
        action='store_true',
        help='Only show file statistics without calling LLM APIs'
    )
    
    parser.add_argument(
        '--specific-model',
        help='Specific model name to use (e.g., gpt-4, deepseek-chat, gemini-1.5-pro)'
    )
    
    parser.add_argument(
        '--output',
        help='Output filename (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-test', '--test-mode',
        action='store_true',
        help='Test mode: show prompts without calling LLM APIs'
    )
    
    parser.add_argument(
        '-verify', '--verify',
        action='store_true',
        help='Verify file consistency: check label column (0/1) and text format (one <group> per text)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Check if we need model for processing (not just statistics, test, or verify mode)
    if not args.statistics_only and not args.test_mode and not args.verify and not args.model:
        print("Error: Model must be specified when not using -stat, -test, or -verify mode")
        print("Use -model option, -stat for statistics only, -test for test mode, or -verify for file verification")
        sys.exit(1)
    
    # Validate limits
    if args.max_rows_per_sheet < 1:
        print("Error: max-rows-per-sheet must be at least 1")
        sys.exit(1)
    
    if args.max_total_rows < 1:
        print("Error: max-total-rows must be at least 1")
        sys.exit(1)


def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)
    
    print("=== Euphemism Analysis Tool ===")
    print(f"Input file: {args.input_file}")
    
    # Verification mode
    if args.verify:
        print("Mode: File verification")
        processor = DataProcessor(args.input_file)
        results = processor.verify_file()
        
        print("\n=== Verification Results ===")
        print(f"File: {args.input_file}")
        print(f"Status: {'✓ VALID' if results['valid'] else '✗ INVALID'}")
        print(f"Sheets: {results['sheets_processed']}/{results['total_sheets']} processed")
        print(f"Total rows: {results['total_rows']}")
        
        if results['errors']:
            print(f"\n✗ ERRORS FOUND ({len(results['errors'])}):")
            print("=" * 80)
            for i, error in enumerate(results['errors'], 1):
                print(f"{i:3d}. {error}")
        
        if results['warnings']:
            print(f"\n⚠ WARNINGS ({len(results['warnings'])}):")
            print("-" * 80)
            for i, warning in enumerate(results['warnings'], 1):
                print(f"{i:3d}. {warning}")
        
        if results['valid']:
            print("\n✓ File is valid and ready for processing!")
        else:
            print("\n✗ File has errors that need to be fixed before processing.")
            sys.exit(1)
        return
    
    # Statistics-only mode
    if args.statistics_only:
        print("Mode: Statistics only")
        stats = get_file_statistics(args.input_file)
        print_statistics(stats)
        return
    
    # Test mode
    if args.test_mode:
        print("Mode: Test mode (showing prompts without LLM calls)")
        print(f"Limits: max_rows_per_sheet={args.max_rows_per_sheet}, max_total_rows={args.max_total_rows}")
        
        # Create data processor in test mode (no LLM client needed)
        processor = DataProcessor(args.input_file, llm_client=None, test_mode=True)
        processor.set_limits(args.max_total_rows, args.max_rows_per_sheet, args.batch_size)
        
        # Process the file in test mode
        processor.process_file()
        return
    
    # Processing mode
    print(f"Mode: Processing with {args.model}")
    print(f"Limits: max_rows_per_sheet={args.max_rows_per_sheet}, max_total_rows={args.max_total_rows}")
    
    try:
        # Get API key and create LLM client
        api_key = get_api_key(args.model)
        llm_client = create_llm_client(args.model, api_key, args.specific_model)
        
        # Create data processor
        processor = DataProcessor(args.input_file, llm_client)
        processor.set_limits(args.max_total_rows, args.max_rows_per_sheet, args.batch_size)
        
        # Process the file
        processed_sheets, sheet_metrics = processor.process_file()
        
        # Generate output filename if not specified
        if args.output:
            output_file = args.output
        else:
            method = get_processing_method(args.max_rows_per_sheet, args.max_total_rows)
            model_name = args.specific_model or SUPPORTED_MODELS[args.model]['default_model']
            experiment_num = find_next_experiment_number(args.model, model_name, method)
            output_file = generate_output_filename(args.model, model_name, method, experiment_num)
        
        # Save results
        processor.save_results(processed_sheets, output_file, sheet_metrics)
        
        print(f"\n=== Processing Complete ===")
        print(f"Total rows processed: {processor.processed_rows}")
        print(f"Output file: {output_file}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()