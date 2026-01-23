#!/usr/bin/env python3
"""
Finetune MamayLM model using data from all sheets and optionally evaluate it.

This script:
1. Loads data from all sheets of PETs_Ukr.xlsx with training phrases containing words in angular brackets
2. Splits each sheet's data 50/25/25 for train/validation/test
3. Combines all training, validation, and test data from all sheets
4. Finetunes MamayLM using LoRA/PEFT with validation tracking
5. Saves the finetuned model
6. Optionally evaluates the finetuned model on the test set

Usage:
    python finetune_mamaylm.py                          # Just finetune
    python finetune_mamaylm.py --evaluate               # Finetune and evaluate
    python finetune_mamaylm.py --eval-only              # Only evaluate existing model
    python finetune_mamaylm.py --evaluate --show-all-queries  # Show all query results
    python finetune_mamaylm.py --predict "Text with <word>" --model-path ./path  # Predict single phrase
"""

import argparse
import pandas as pd
import numpy as np
import torch
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset
from config import SYSTEM_PROMPT

# Configuration
MODEL_NAME = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
OUTPUT_DIR = "./mamaylm_finetuned"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
MAX_LENGTH = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100


def format_prompt(text: str, label: int = None) -> str:
    """Format the prompt for finetuning or inference."""
    user_prompt = f"Text: {text}"
    
    if label is not None:
        # Training format with label
        s = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant: {label}"
        print(s)
        return s
    else:
        # Inference format without label
        return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"


def load_and_split_data(xlsx_path: str = "PETs_Ukr.xlsx"):
    """Load data from all sheets of PETs_Ukr.xlsx and split each sheet 50/25/25 for train/val/test.
    
    The training phrases contain the word/phrase in angular brackets (e.g., <word>)
    as specified in the 'text' column of PETs_Ukr.xlsx.
    """
    print(f"Loading data from all sheets in {xlsx_path}...")
    
    # Load all sheets
    xl = pd.ExcelFile(xlsx_path)
    sheet_names = xl.sheet_names
    print(f"Found {len(sheet_names)} sheets: {sheet_names}")
    
    # Lists to collect all training, validation, and test data
    all_train_texts = []
    all_train_labels = []
    all_val_texts = []
    all_val_labels = []
    all_test_texts = []
    all_test_labels = []
    all_test_categories = []
    
    # Process each sheet separately
    for sheet_name in sheet_names:
        print(f"\nProcessing sheet: {sheet_name}")
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        print(f"  Loaded {len(df)} examples from '{sheet_name}'")
        
        # Get text (with angular brackets) and labels
        texts = df['text'].values
        labels = df['label'].values
        # Use sheet name as category (some sheets don't have 'category' column)
        if 'category' in df.columns:
            categories = df['category'].values
        else:
            categories = np.array([sheet_name] * len(df))
        
        print(f"  Label distribution: {dict(pd.Series(labels).value_counts())}")
        
        # Split data: 50% for training, 25% for validation, 25% for testing
        # Use stratify to maintain label balance in all sets
        try:
            # First split: 50% train, 50% temp (which will be split into val and test)
            train_texts, temp_texts, train_labels, temp_labels, train_idx, temp_idx = train_test_split(
                texts, labels, np.arange(len(texts)), 
                test_size=0.5, random_state=42, stratify=labels
            )
            # Second split: split temp into 50% val and 50% test (25% and 25% of original)
            val_texts, test_texts, val_labels, test_labels, val_idx, test_idx = train_test_split(
                temp_texts, temp_labels, temp_idx,
                test_size=0.5, random_state=42, stratify=temp_labels
            )
        except ValueError:
            # If stratification fails (e.g., only one class), split without stratification
            print(f"  Warning: Cannot stratify sheet '{sheet_name}', using simple split")
            train_texts, temp_texts, train_labels, temp_labels, train_idx, temp_idx = train_test_split(
                texts, labels, np.arange(len(texts)), 
                test_size=0.5, random_state=42
            )
            val_texts, test_texts, val_labels, test_labels, val_idx, test_idx = train_test_split(
                temp_texts, temp_labels, temp_idx,
                test_size=0.5, random_state=42
            )
        
        test_categories = categories[test_idx]
        
        print(f"  Training: {len(train_texts)} examples, Validation: {len(val_texts)} examples, Test: {len(test_texts)} examples")
        
        # Add to combined lists
        all_train_texts.extend(train_texts)
        all_train_labels.extend(train_labels)
        all_val_texts.extend(val_texts)
        all_val_labels.extend(val_labels)
        all_test_texts.extend(test_texts)
        all_test_labels.extend(test_labels)
        all_test_categories.extend(test_categories)
    
    # Convert to numpy arrays
    all_train_texts = np.array(all_train_texts)
    all_train_labels = np.array(all_train_labels)
    all_val_texts = np.array(all_val_texts)
    all_val_labels = np.array(all_val_labels)
    all_test_texts = np.array(all_test_texts)
    all_test_labels = np.array(all_test_labels)
    all_test_categories = np.array(all_test_categories)
    
    print("\n" + "="*80)
    print("COMBINED DATASET STATISTICS")
    print("="*80)
    print(f"Total training examples: {len(all_train_texts)}")
    print(f"Total validation examples: {len(all_val_texts)}")
    print(f"Total test examples: {len(all_test_texts)}")
    print(f"Training label distribution: {dict(pd.Series(all_train_labels).value_counts())}")
    print(f"Validation label distribution: {dict(pd.Series(all_val_labels).value_counts())}")
    print(f"Test label distribution: {dict(pd.Series(all_test_labels).value_counts())}")
    print(f"Test categories: {sorted(np.unique(all_test_categories))}")
    print("="*80)
    
    return all_train_texts, all_train_labels, all_val_texts, all_val_labels, all_test_texts, all_test_labels, all_test_categories


def prepare_dataset(texts, labels, tokenizer):
    """Prepare dataset for finetuning with memory-efficient processing."""
    print("Preparing dataset...")
    
    # Format prompts one at a time to save memory
    input_ids_list = []
    attention_mask_list = []
    
    # Process in small batches
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Format prompts for this batch
        formatted_batch = [format_prompt(text, label) for text, label in zip(batch_texts, batch_labels)]
        
        # Tokenize this batch
        tokenized = tokenizer(
            formatted_batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None
        )
        
        input_ids_list.extend(tokenized['input_ids'])
        attention_mask_list.extend(tokenized['attention_mask'])
        
        # Clear batch from memory
        del formatted_batch, tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list
    })
    
    # Clear temporary lists
    del input_ids_list, attention_mask_list
    gc.collect()
    
    print(f"Dataset prepared with {len(dataset)} examples")
    return dataset


def load_base_model(model_name: str):
    """Load the base MamayLM model with 8-bit quantization."""
    print(f"Loading base model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to use 8-bit quantization if available, otherwise load in bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("Model loaded with 8-bit quantization")
    except Exception as e:
        print(f"8-bit quantization not available ({e}), loading in bfloat16...")
        # Load without max_memory to avoid offloading to meta device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",  # Force to single GPU
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    
    print("Base model loaded successfully!")
    return tokenizer, model


def setup_lora(model):
    """Set up LoRA configuration for efficient finetuning."""
    print("Setting up LoRA...")
    
    # Prepare model for k-bit training
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    
    # For Gemma3, target all attention projection layers and MLP layers
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Verify trainable parameters exist
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters found after applying LoRA!")
    
    return model


def finetune_model(train_texts, train_labels, val_texts, val_labels, output_dir: str = OUTPUT_DIR):
    """Finetune MamayLM using LoRA with validation."""
    print("\n" + "="*80)
    print("STARTING FINETUNING")
    print("="*80)
    
    # Load model and tokenizer
    tokenizer, model = load_base_model(MODEL_NAME)
    
    # Apply LoRA
    model = setup_lora(model)
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_texts, train_labels, tokenizer)
    val_dataset = prepare_dataset(val_texts, val_labels, tokenizer)
    
    # Training arguments with memory optimization and validation
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=False,  # Disable to avoid conflicts with PEFT
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    print(f"\nSaving finetuned model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Clean up training resources
    del trainer
    del model
    del train_dataset
    del val_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Finetuning complete!")
    print("="*80)


def load_finetuned_model(model_path: str = OUTPUT_DIR):
    """Load the finetuned model."""
    print(f"Loading finetuned model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Merge LoRA weights into base model
    
    print("Finetuned model loaded successfully!")
    return tokenizer, model


def predict_single(text: str, tokenizer, model) -> int:
    """Make a prediction for a single text with memory optimization."""
    prompt = format_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = result[len(prompt):].strip()
    
    # Clean up tensors immediately
    del inputs, outputs
    
    # Extract label (0 or 1) from result
    for char in result:
        if char == '1':
            return 1
        elif char == '0':
            return 0
    
    return 0  # Default to 0 if no clear label


def evaluate_model(test_texts, test_labels, test_categories, model_path: str = OUTPUT_DIR, show_all_queries: bool = False):
    """Evaluate the finetuned model on test set."""
    print("\n" + "="*80)
    print("EVALUATING FINETUNED MODEL")
    print("="*80)
    
    # Load finetuned model
    tokenizer, model = load_finetuned_model(model_path)
    model.eval()
    
    # Make predictions with aggressive memory management
    print(f"\nMaking predictions on {len(test_texts)} test examples...")
    predictions = []
    
    for i, text in enumerate(test_texts):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_texts)} examples...")
        
        pred = predict_single(text, tokenizer, model)
        predictions.append(pred)
        
        # Show query results if requested
        if show_all_queries:
            correct = "✓" if pred == test_labels[i] else "✗"
            print(f"  [{i+1}] {correct} Text: {text[:80]}... | Predicted: {pred} | Actual: {test_labels[i]}")
        
        # Aggressive memory cleanup (more frequent)
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    predictions = np.array(predictions)
    
    # Overall metrics
    print("\n" + "="*80)
    print("OVERALL TEST SET PERFORMANCE")
    print("="*80)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary', zero_division=0
    )
    cm = confusion_matrix(test_labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")
    
    # Per-category statistics
    print("\n" + "="*80)
    print("PER-CATEGORY STATISTICS")
    print("="*80)
    
    category_stats = []
    for category_name in sorted(np.unique(test_categories)):
        mask = test_categories == category_name
        
        if mask.sum() == 0:
            continue
        
        category_labels = test_labels[mask]
        category_preds = predictions[mask]
        
        n_samples = mask.sum()
        n_label_0 = (category_labels == 0).sum()
        n_label_1 = (category_labels == 1).sum()
        
        acc = accuracy_score(category_labels, category_preds)
        prec, rec, f1_s, _ = precision_recall_fscore_support(
            category_labels, category_preds, average='binary', zero_division=0
        )
        cm_category = confusion_matrix(category_labels, category_preds, labels=[0, 1])
        
        tp = cm_category[1,1]
        tn = cm_category[0,0]
        fp = cm_category[0,1]
        fn = cm_category[1,0]
        
        print(f"\n{category_name.upper()}:")
        print(f"  Samples: {n_samples} (Label 0: {n_label_0}, Label 1: {n_label_1})")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%) | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1_s:.4f}")
        print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
        
        category_stats.append({
            'category': category_name,
            'n_samples': n_samples,
            'n_label_0': n_label_0,
            'n_label_1': n_label_1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1_s,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        })
    
    # Save statistics
    stats_df = pd.DataFrame(category_stats)
    stats_file = 'mamaylm_finetuned_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"\n\nPer-category statistics saved to {stats_file}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average Accuracy: {stats_df['accuracy'].mean():.4f}")
    print(f"Average F1 Score: {stats_df['f1_score'].mean():.4f}")
    print(f"Average Precision: {stats_df['precision'].mean():.4f}")
    print(f"Average Recall: {stats_df['recall'].mean():.4f}")
    print("="*80)
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Finetune MamayLM using half of the data with optional evaluation"
    )
    parser.add_argument(
        '--evaluate', 
        action='store_true',
        help='Evaluate the model after finetuning'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate existing finetuned model (skip training)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=OUTPUT_DIR,
        help=f'Path to finetuned model (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='PETs_Ukr.xlsx',
        help='Path to data file (default: PETs_Ukr.xlsx)'
    )
    parser.add_argument(
        '--show-all-queries',
        action='store_true',
        help='Show all query results during evaluation'
    )
    parser.add_argument(
        '--predict',
        type=str,
        default=None,
        help='Predict classification for a single phrase (e.g., "Text with <word> in brackets")'
    )
    
    args = parser.parse_args()
    
    # Single phrase prediction mode
    if args.predict:
        print("\n" + "="*80)
        print("SINGLE PHRASE PREDICTION")
        print("="*80)
        print(f"Text: {args.predict}")
        print(f"Model: {args.model_path}")
        print()
        
        # Load finetuned model
        tokenizer, model = load_finetuned_model(args.model_path)
        model.eval()
        
        # Make prediction
        prediction = predict_single(args.predict, tokenizer, model)
        
        print("\n" + "="*80)
        print(f"Prediction: {prediction}")
        print(f"Classification: {'Euphemism (1)' if prediction == 1 else 'Not euphemism (0)'}")
        print("="*80)
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        print("\nDone!")
        return
    
    # Load and split data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, test_categories = load_and_split_data(args.data_path)
    
    # Finetune model (unless eval-only)
    if not args.eval_only:
        finetune_model(train_texts, train_labels, val_texts, val_labels, args.model_path)
    else:
        print("Skipping finetuning (--eval-only mode)")
    
    # Evaluate model if requested or if eval-only
    if args.evaluate or args.eval_only:
        evaluate_model(test_texts, test_labels, test_categories, args.model_path, args.show_all_queries)
    else:
        print("\nSkipping evaluation. Use --evaluate flag to evaluate the model.")
        print(f"To evaluate later, run: python {__file__} --eval-only")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
