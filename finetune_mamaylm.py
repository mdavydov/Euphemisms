#!/usr/bin/env python3
"""
Finetune MamayLM model using data from all sheets and optionally evaluate it.

Supports two finetuning modes:
  1. LoRA (default) – finetunes low-rank adapters on attention/MLP weights
  2. Prompt tuning (--prompt-tuning) – finetunes only the system-prompt
     embeddings while keeping all network weights frozen

This script:
1. Loads data from all sheets of PETs_Ukr.xlsx with training phrases containing words in angular brackets
2. Splits each sheet's data 50/25/25 for train/validation/test
3. Combines all training, validation, and test data from all sheets
4. Finetunes MamayLM using LoRA/PEFT or prompt tuning with validation tracking
5. Saves the finetuned model
6. Optionally evaluates the finetuned model on the test set

Usage:
    python finetune_mamaylm.py                          # LoRA finetune
    python finetune_mamaylm.py --prompt-tuning          # Prompt-tuning finetune
    python finetune_mamaylm.py --evaluate               # Finetune and evaluate
    python finetune_mamaylm.py --eval-only              # Only evaluate existing model
    python finetune_mamaylm.py --prompt-tuning --eval-only  # Evaluate prompt-tuned model
    python finetune_mamaylm.py --predict "Text with <word>" --model-path ./path
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
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
    PeftModel,
    TaskType,
)
from datasets import Dataset
from config import SYSTEM_PROMPT

# Configuration
MODEL_NAME = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
OUTPUT_DIR = "./mamaylm_finetuned"
PROMPT_TUNING_OUTPUT_DIR = "./mamaylm_prompt_tuned"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
MAX_LENGTH = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
PROMPT_TUNING_LEARNING_RATE = 3e-2  # Higher LR typical for prompt tuning
NUM_EPOCHS = 3
WARMUP_STEPS = 100


def format_prompt(text: str, label: int = None, include_system_prompt: bool = True) -> str:
    """Format the prompt for finetuning or inference.

    Args:
        text: The input text.
        label: Ground-truth label (0/1) for training; None for inference.
        include_system_prompt: If False, omit the system prompt (used for
            prompt-tuning where learned embeddings replace it).
    """
    user_prompt = f"Text: {text}"
    prefix = SYSTEM_PROMPT if include_system_prompt else ""

    if label is not None:
        # Training format with label
        s = f"{prefix}\n\nUser: {user_prompt}\nAssistant: {label}"
        print(s)
        return s
    else:
        # Inference format without label
        return f"{prefix}\n\nUser: {user_prompt}\nAssistant:"


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
        try:
            train_texts, temp_texts, train_labels, temp_labels, train_idx, temp_idx = train_test_split(
                texts, labels, np.arange(len(texts)),
                test_size=0.5, random_state=42, stratify=labels
            )
            val_texts, test_texts, val_labels, test_labels, val_idx, test_idx = train_test_split(
                temp_texts, temp_labels, temp_idx,
                test_size=0.5, random_state=42, stratify=temp_labels
            )
        except ValueError:
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


def prepare_dataset(texts, labels, tokenizer, include_system_prompt: bool = True):
    """Prepare dataset for finetuning with memory-efficient processing."""
    print("Preparing dataset...")

    input_ids_list = []
    attention_mask_list = []

    # Process in small batches
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        # Format prompts for this batch
        formatted_batch = [
            format_prompt(text, label, include_system_prompt=include_system_prompt)
            for text, label in zip(batch_texts, batch_labels)
        ]

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

        del formatted_batch, tokenized

    # Create token_type_ids (all zeros, required by Gemma3)
    token_type_ids_list = [[0] * len(ids) for ids in input_ids_list]

    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list
    })

    del input_ids_list, attention_mask_list, token_type_ids_list
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    print("Base model loaded successfully!")
    return tokenizer, model


def setup_lora(model):
    """Set up LoRA configuration for efficient finetuning."""
    print("Setting up LoRA...")

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters found after applying LoRA!")

    return model


def patch_peft_forward_for_gemma3(peft_model, num_virtual_tokens):
    """Monkey-patch PEFT model forward to pass token_type_ids through for Gemma3.

    PEFT strips token_type_ids but Gemma3 requires them during training.
    This patch extends token_type_ids with zeros for virtual tokens
    (analogous to how PEFT extends attention_mask) and passes them through.
    """
    import types as _types

    def _patched_forward(self, input_ids=None, attention_mask=None,
                         inputs_embeds=None, labels=None,
                         output_attentions=None, output_hidden_states=None,
                         return_dict=None, task_ids=None, **kwargs):
        batch_size = (input_ids.shape[0] if input_ids is not None
                      else inputs_embeds.shape[0])

        # Pop token_type_ids before PEFT can strip it with a warning
        token_type_ids = kwargs.pop("token_type_ids", None)

        # Get prompt embeddings from PEFT
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)

        # Convert input_ids -> inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Prepend virtual-token embeddings
        inputs_embeds = torch.cat(
            (prompts.to(inputs_embeds.dtype), inputs_embeds), dim=1)

        # Extend attention_mask
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(
                batch_size, num_virtual_tokens,
                device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)

        # Extend token_type_ids (required by Gemma3)
        if token_type_ids is not None:
            prefix_token_type_ids = torch.zeros(
                batch_size, num_virtual_tokens,
                device=token_type_ids.device, dtype=token_type_ids.dtype)
            kwargs["token_type_ids"] = torch.cat(
                (prefix_token_type_ids, token_type_ids), dim=1)

        # Extend labels
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, num_virtual_tokens), -100,
                device=labels.device, dtype=labels.dtype)
            labels = torch.cat((prefix_labels, labels), dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds, labels=labels,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, **kwargs)

    peft_model.forward = _types.MethodType(_patched_forward, peft_model)
    print("Patched PEFT forward to pass token_type_ids for Gemma3.")


def setup_prompt_tuning(model, tokenizer):
    """Set up prompt tuning: freeze all model weights, learn only prompt embeddings.

    The learned embeddings are initialised from the tokenised SYSTEM_PROMPT so
    that the optimisation starts from a meaningful point.  During training only
    these embeddings are updated – the rest of the network is frozen.
    """
    print("Setting up prompt tuning...")

    # Determine number of virtual tokens from the system prompt length
    system_tokens = tokenizer(SYSTEM_PROMPT, return_tensors="pt").input_ids
    num_virtual_tokens = system_tokens.shape[1]
    print(f"System prompt tokenises to {num_virtual_tokens} tokens → "
          f"using {num_virtual_tokens} virtual tokens for prompt tuning")

    pt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=SYSTEM_PROMPT,
        num_virtual_tokens=num_virtual_tokens,
        tokenizer_name_or_path=MODEL_NAME,
    )

    model = get_peft_model(model, pt_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (prompt embeddings only): {trainable_params:,}")
    if trainable_params == 0:
        raise ValueError("No trainable parameters found after setting up prompt tuning!")

    return model, num_virtual_tokens


def extract_and_save_prompt_embeddings(model, output_path: str):
    """Extract the learned prompt embeddings from a PEFT prompt-tuning model
    and save them as a standalone .pt tensor file.

    This file can later be loaded by run_query.py via --prompt-embeds.
    """
    # PEFT stores prompt embeddings in prompt_encoder.default.embedding.weight
    try:
        embedding_weight = model.prompt_encoder["default"].embedding.weight.data
        torch.save(embedding_weight.cpu(), str(output_path))
        print(f"Saved prompt embeddings to {output_path}: shape {embedding_weight.shape}")
        return
    except (AttributeError, KeyError):
        pass

    # Fallback: search named parameters
    for name, param in model.named_parameters():
        if "prompt_embeddings" in name or "prompt_encoder" in name:
            torch.save(param.data.cpu(), str(output_path))
            print(f"Saved prompt embeddings to {output_path}: shape {param.data.shape}")
            return

    print("Warning: Could not find prompt embeddings in model parameters")


def finetune_model(train_texts, train_labels, val_texts, val_labels,
                   output_dir: str = OUTPUT_DIR, prompt_tuning: bool = False):
    """Finetune MamayLM using LoRA (default) or prompt tuning."""
    mode_label = "PROMPT TUNING" if prompt_tuning else "LoRA"
    print("\n" + "="*80)
    print(f"STARTING FINETUNING ({mode_label})")
    print("="*80)

    # Load model and tokenizer
    tokenizer, model = load_base_model(MODEL_NAME)

    # Apply the chosen parameter-efficient method
    if prompt_tuning:
        model, num_virtual_tokens = setup_prompt_tuning(model, tokenizer)
        patch_peft_forward_for_gemma3(model, num_virtual_tokens)
        include_system_prompt = False  # learned embeddings replace system prompt
        lr = PROMPT_TUNING_LEARNING_RATE
    else:
        model = setup_lora(model)
        include_system_prompt = True
        lr = LEARNING_RATE

    # Prepare datasets
    train_dataset = prepare_dataset(train_texts, train_labels, tokenizer,
                                    include_system_prompt=include_system_prompt)
    val_dataset = prepare_dataset(val_texts, val_labels, tokenizer,
                                  include_system_prompt=include_system_prompt)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
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
        gradient_checkpointing=False,
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

    # Save the model (PEFT adapter)
    print(f"\nSaving finetuned model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # For prompt tuning, also save the raw prompt embeddings as a .pt file
    if prompt_tuning:
        pt_path = str(Path(output_dir) / "prompt_embeddings.pt")
        extract_and_save_prompt_embeddings(trainer.model, pt_path)

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
    """Load the finetuned LoRA model."""
    print(f"Loading finetuned model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Load LoRA weights and merge into base model
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()

    print("Finetuned model loaded successfully!")
    return tokenizer, model


def load_prompt_tuned_model(model_path: str = PROMPT_TUNING_OUTPUT_DIR):
    """Load a prompt-tuned model (PEFT adapter kept – cannot be merged)."""
    print(f"Loading prompt-tuned model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Load prompt-tuning adapter (note: cannot merge_and_unload for prompt tuning)
    model = PeftModel.from_pretrained(base_model, model_path)

    print("Prompt-tuned model loaded successfully!")
    return tokenizer, model


def predict_single(text: str, tokenizer, model, prompt_tuning: bool = False) -> int:
    """Make a prediction for a single text with memory optimization."""
    prompt = format_prompt(text, include_system_prompt=not prompt_tuning)
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


def evaluate_model(test_texts, test_labels, test_categories,
                   model_path: str = OUTPUT_DIR,
                   show_all_queries: bool = False,
                   prompt_tuning: bool = False):
    """Evaluate the finetuned model on test set."""
    print("\n" + "="*80)
    mode_label = "PROMPT-TUNED" if prompt_tuning else "FINETUNED"
    print(f"EVALUATING {mode_label} MODEL")
    print("="*80)

    # Load the appropriate model
    if prompt_tuning:
        tokenizer, model = load_prompt_tuned_model(model_path)
    else:
        tokenizer, model = load_finetuned_model(model_path)
    model.eval()

    # Make predictions with aggressive memory management
    print(f"\nMaking predictions on {len(test_texts)} test examples...")
    predictions = []

    for i, text in enumerate(test_texts):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_texts)} examples...")

        pred = predict_single(text, tokenizer, model, prompt_tuning=prompt_tuning)
        predictions.append(pred)

        # Show query results if requested
        if show_all_queries:
            correct = "✓" if pred == test_labels[i] else "✗"
            print(f"  [{i+1}] {correct} Text: {text[:80]}... | Predicted: {pred} | Actual: {test_labels[i]}")

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
    suffix = "_prompt_tuned" if prompt_tuning else "_finetuned"
    stats_file = f'mamaylm{suffix}_statistics.csv'
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
        '--prompt-tuning',
        action='store_true',
        help='Use prompt tuning instead of LoRA: optimise only the system-prompt '
             'embeddings while keeping all network weights frozen'
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
        default=None,
        help='Path to finetuned model (default: auto-selected based on mode)'
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

    # Resolve default model path based on mode
    if args.model_path is None:
        args.model_path = PROMPT_TUNING_OUTPUT_DIR if args.prompt_tuning else OUTPUT_DIR

    # Single phrase prediction mode
    if args.predict:
        print("\n" + "="*80)
        print("SINGLE PHRASE PREDICTION")
        print("="*80)
        print(f"Text: {args.predict}")
        print(f"Model: {args.model_path}")
        print(f"Mode: {'prompt tuning' if args.prompt_tuning else 'LoRA'}")
        print()

        # Load model
        if args.prompt_tuning:
            tokenizer, model = load_prompt_tuned_model(args.model_path)
        else:
            tokenizer, model = load_finetuned_model(args.model_path)
        model.eval()

        # Make prediction
        prediction = predict_single(args.predict, tokenizer, model,
                                    prompt_tuning=args.prompt_tuning)

        print("\n" + "="*80)
        print(f"Prediction: {prediction}")
        print(f"Classification: {'Euphemism (1)' if prediction == 1 else 'Not euphemism (0)'}")
        print("="*80)

        del model, tokenizer
        torch.cuda.empty_cache()
        print("\nDone!")
        return

    # Load and split data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, test_categories = load_and_split_data(args.data_path)

    # Finetune model (unless eval-only)
    if not args.eval_only:
        finetune_model(train_texts, train_labels, val_texts, val_labels,
                       args.model_path, prompt_tuning=args.prompt_tuning)
    else:
        print("Skipping finetuning (--eval-only mode)")

    # Evaluate model if requested or if eval-only
    if args.evaluate or args.eval_only:
        evaluate_model(test_texts, test_labels, test_categories,
                       args.model_path, args.show_all_queries,
                       prompt_tuning=args.prompt_tuning)
    else:
        print("\nSkipping evaluation. Use --evaluate flag to evaluate the model.")
        print(f"To evaluate later, run: python {__file__} --eval-only")

    print("\nDone!")


if __name__ == '__main__':
    main()
