#!/usr/bin/env python3
"""
Run MamayLM with custom queries using base or finetuned model.

Usage:
    python run_query.py "Your query here"                    # Use base model
    python run_query.py "Your query" --model-path ./path     # Use finetuned model
    python run_query.py "Text: <word> phrase" --classify     # Classification format
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
from config import SYSTEM_PROMPT

# Configuration
BASE_MODEL_NAME = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
DEFAULT_OUTPUT_DIR = "./mamaylm_finetuned"


def format_prompt(text: str, use_system_prompt: bool = False) -> str:
    """Format the prompt for inference."""
    if use_system_prompt:
        user_prompt = f"Text: {text}"
        return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"
    else:
        return text


def load_base_model(model_name: str = BASE_MODEL_NAME):
    """Load the base MamayLM model."""
    print(f"Loading base model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    print("Base model loaded successfully!")
    return tokenizer, model


def load_finetuned_model(model_path: str):
    """Load the finetuned model with LoRA weights."""
    print(f"Loading finetuned model from {model_path}...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load tokenizer from finetuned path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
    
    print("Finetuned model loaded successfully!")
    return tokenizer, model


def run_query(query: str, tokenizer, model, max_new_tokens: int = 10) -> str:
    """Run a single query through the model."""
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(query):].strip()
    
    return response


def extract_classification(response: str) -> int:
    """Extract binary classification (0 or 1) from response."""
    for char in response:
        if char == '1':
            return 1
        elif char == '0':
            return 0
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run queries using base or finetuned MamayLM model"
    )
    parser.add_argument(
        'query',
        type=str,
        nargs='?',
        help='Query text to run through the model'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help=f'Path to finetuned model (default: use base model)'
    )
    parser.add_argument(
        '--classify',
        action='store_true',
        help='Use classification format with system prompt (for finetuned model)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=10,
        help='Maximum number of new tokens to generate (default: 10)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode for multiple queries'
    )
    
    args = parser.parse_args()
    
    # Load model
    if args.model_path:
        tokenizer, model = load_finetuned_model(args.model_path)
    else:
        tokenizer, model = load_base_model()
    
    model.eval()
    print()
    
    # Interactive mode
    if args.interactive:
        print("Interactive mode. Type 'quit' or 'exit' to stop.")
        print("=" * 80)
        
        while True:
            try:
                query_input = input("\nQuery: ").strip()
                
                if query_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break
                
                if not query_input:
                    continue
                
                # Format query if classification mode
                query = format_prompt(query_input, args.classify) if args.classify else query_input
                
                # Run query
                response = run_query(query, tokenizer, model, args.max_tokens)
                
                # Display results
                if args.classify:
                    classification = extract_classification(response)
                    if classification is not None:
                        print(f"Classification: {classification}")
                    print(f"Raw response: {response}")
                else:
                    print(f"Response: {response}")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Single query mode
    else:
        if not args.query:
            parser.error("Query text is required in non-interactive mode")
        
        query_input = args.query
        
        # Format query if classification mode
        query = format_prompt(query_input, args.classify) if args.classify else query_input
        
        print(f"Query: {query_input}")
        print("=" * 80)
        
        # Run query
        response = run_query(query, tokenizer, model, args.max_tokens)
        
        # Display results
        print(f"\nResponse: {response}")
        
        if args.classify:
            classification = extract_classification(response)
            if classification is not None:
                print(f"Classification: {classification}")
            else:
                print("Classification: Could not extract (0 or 1)")
        
        print()


if __name__ == "__main__":
    main()
