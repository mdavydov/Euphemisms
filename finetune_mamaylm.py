#!/usr/bin/env python3
"""
Finetune MamayLM model using data from all sheets and optionally evaluate it.

Supports three finetuning modes (one is required):
  --lora          – finetunes low-rank adapters on attention/MLP weights
  --prompt-tuning – finetunes only the system-prompt embeddings; all weights frozen
  --full-finetune – updates all model parameters, no PEFT

Evaluation model sources:
  --lora          – load fine-tuned LoRA model
  --prompt-tuning – load prompt-tuned PEFT adapter
  --full-finetune – load fully fine-tuned model
  --original      – load original base model (no fine-tuning; skips mode requirement)

This script:
1. Loads training data from PETs_Ukr_Train.xlsx (split 80/20 into train/validation)
   only when finetuning (skipped in --eval-only mode)
2. Loads test data from PETs_Ukr_Test.xlsx only when evaluating
3. Finetunes MamayLM using the chosen method with validation tracking
4. Saves the finetuned model
5. Optionally evaluates the model on the test set

Usage:
    python finetune_mamaylm.py --lora                             # LoRA fine-tune
    python finetune_mamaylm.py --prompt-tuning                    # Prompt-tuning fine-tune
    python finetune_mamaylm.py --full-finetune                    # Full fine-tuning (all params)
    python finetune_mamaylm.py --lora --evaluate                  # Fine-tune and evaluate
    python finetune_mamaylm.py --lora --eval-only                 # Evaluate fine-tuned LoRA model
    python finetune_mamaylm.py --eval-only --original             # Evaluate original base model
    python finetune_mamaylm.py --full-finetune --eval-only        # Evaluate fully fine-tuned model
    python finetune_mamaylm.py --prompt-tuning --eval-only        # Evaluate prompt-tuned model
    python finetune_mamaylm.py --lora --predict "Text with <word>" --model-path ./path
    python finetune_mamaylm.py --lora --iterations 3              # 3 iterations, saves finetune-01..03
    python finetune_mamaylm.py --lora --iterations 2 --resume-from finetune-03  # continue from iter 3
"""

import argparse
import math
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
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import Dataset
from config import SYSTEM_PROMPT

# Configuration
MODEL_NAME = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
OUTPUT_DIR = "./mamaylm_finetuned"
PROMPT_TUNING_OUTPUT_DIR = "./mamaylm_prompt_tuned"
FULL_FINETUNE_OUTPUT_DIR = "./mamaylm_full_finetuned"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
MAX_LENGTH = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
PROMPT_TUNING_LEARNING_RATE = 3e-2  # Higher LR typical for prompt tuning
FULL_FINETUNE_LEARNING_RATE = 2e-5  # Lower LR for full fine-tuning
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


def load_train_val_data(train_path: str = "PETs_Ukr_Train.xlsx"):
    """Load training data from an xlsx file and split 80/20 into train/validation.

    The training phrases contain the word/phrase in angular brackets (e.g., <word>)
    as specified in the 'text' column.
    """
    print(f"Loading training data from {train_path}...")
    xl_train = pd.ExcelFile(train_path)
    print(f"Found {len(xl_train.sheet_names)} sheets: {xl_train.sheet_names}")

    all_train_texts = []
    all_train_labels = []
    all_val_texts = []
    all_val_labels = []

    for sheet_name in xl_train.sheet_names:
        print(f"\nProcessing training sheet: {sheet_name}")
        df = pd.read_excel(train_path, sheet_name=sheet_name)
        print(f"  Loaded {len(df)} examples from '{sheet_name}'")

        texts = df['text'].values
        labels = df['label'].values
        print(f"  Label distribution: {dict(pd.Series(labels).value_counts())}")

        # Split: 80% train, 20% validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=0.2, random_state=42
        )

        print(f"  Training: {len(train_texts)} examples, Validation: {len(val_texts)} examples")

        all_train_texts.extend(train_texts)
        all_train_labels.extend(train_labels)
        all_val_texts.extend(val_texts)
        all_val_labels.extend(val_labels)

    all_train_texts = np.array(all_train_texts)
    all_train_labels = np.array(all_train_labels)
    all_val_texts = np.array(all_val_texts)
    all_val_labels = np.array(all_val_labels)

    print("\n" + "="*80)
    print("TRAINING DATASET STATISTICS")
    print("="*80)
    print(f"Total training examples:   {len(all_train_texts)}")
    print(f"Total validation examples: {len(all_val_texts)}")
    print(f"Training label distribution:   {dict(pd.Series(all_train_labels).value_counts())}")
    print(f"Validation label distribution: {dict(pd.Series(all_val_labels).value_counts())}")
    print("="*80)

    return all_train_texts, all_train_labels, all_val_texts, all_val_labels


def load_test_data(test_path: str = "PETs_Ukr_Test.xlsx"):
    """Load test data from an xlsx file for evaluation.

    Returns:
        Tuple of (texts, labels, sheet_names) where sheet_names tracks the
        originating sheet for every sample so that per-sheet metrics can be
        computed identically to process.py.
    """
    print(f"\nLoading test data from {test_path}...")
    xl_test = pd.ExcelFile(test_path)
    print(f"Found {len(xl_test.sheet_names)} sheets: {xl_test.sheet_names}")

    all_test_texts = []
    all_test_labels = []
    all_test_sheet_names = []

    for sheet_name in xl_test.sheet_names:
        print(f"\nProcessing test sheet: {sheet_name}")
        df = pd.read_excel(test_path, sheet_name=sheet_name)
        print(f"  Loaded {len(df)} examples from '{sheet_name}'")

        texts = df['text'].values
        labels = df['label'].values

        print(f"  Label distribution: {dict(pd.Series(labels).value_counts())}")

        all_test_texts.extend(texts)
        all_test_labels.extend(labels)
        all_test_sheet_names.extend([sheet_name] * len(df))

    all_test_texts = np.array(all_test_texts)
    all_test_labels = np.array(all_test_labels)
    all_test_sheet_names = np.array(all_test_sheet_names)

    print("\n" + "="*80)
    print("TEST DATASET STATISTICS")
    print("="*80)
    print(f"Total test examples: {len(all_test_texts)}")
    print(f"Test label distribution: {dict(pd.Series(all_test_labels).value_counts())}")
    print(f"Test sheets: {list(dict.fromkeys(all_test_sheet_names))}")
    print("="*80)

    return all_test_texts, all_test_labels, all_test_sheet_names


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

    # Disable KV-caching for training/eval forward passes: with teacher-forced
    # full-sequence inputs the cache isn't used, and leaving it enabled makes
    # Gemma3 return a DynamicCache object in the output tuple, which crashes
    # Trainer's cross-process padding/gathering during evaluation.
    model.config.use_cache = False

    print("Base model loaded successfully!")
    return tokenizer, model


def setup_lora(model):
    """Set up LoRA configuration for efficient finetuning."""
    print("Setting up LoRA...")

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


def _preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax token IDs to save memory during evaluation."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _compute_token_accuracy(eval_preds):
    """Compute token-level accuracy for causal LM evaluation.

    For causal LM the prediction at position *i* predicts token *i+1*,
    so we shift predictions and labels before comparing.  Padding
    positions (label == -100) are ignored.
    """
    preds, labels = eval_preds
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100
    correct = (preds[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = float(correct) / float(total) if total > 0 else 0.0
    return {'accuracy': round(accuracy, 4)}


class _EpochTrainEvalCallback(TrainerCallback):
    """Evaluates on the *training* set at each epoch end so that training
    accuracy is available alongside the standard validation metrics."""

    def __init__(self):
        self.train_metrics_per_epoch = []
        self._trainer = None

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._trainer is None:
            return
        epoch = int(state.epoch)
        print(f"\n  Evaluating on training set (epoch {epoch})...")
        metrics = self._trainer.evaluate(
            self._trainer.train_dataset, metric_key_prefix="train",
            ignore_keys=["past_key_values"],
        )
        self.train_metrics_per_epoch.append({'epoch': epoch, **metrics})


def _save_training_stats(trainer, epoch_callback, suffix, output_dir='.'):
    """Extract per-epoch statistics from the trainer log history and the
    epoch callback, then write a semicolon-separated CSV.

    Columns: epoch;train_loss;train_accuracy;eval_loss;eval_accuracy
    """
    log_history = trainer.state.log_history

    # Average training loss per epoch from step-level logs
    epoch_train_losses = {}
    for entry in log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            ep = math.ceil(entry['epoch'])
            epoch_train_losses.setdefault(ep, []).append(entry['loss'])

    # Per-epoch eval metrics (from eval_strategy="epoch")
    eval_per_epoch = {}
    for entry in log_history:
        if 'eval_loss' in entry:
            eval_per_epoch[int(entry['epoch'])] = entry

    # Per-epoch training-set eval metrics from callback
    train_eval_per_epoch = {}
    for entry in epoch_callback.train_metrics_per_epoch:
        train_eval_per_epoch[entry['epoch']] = entry

    all_epochs = sorted(set(epoch_train_losses) | set(eval_per_epoch))

    stats_file = str(Path(output_dir) / f'mamaylm{suffix}_training_stats.csv')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write('epoch;train_loss;train_accuracy;eval_loss;eval_accuracy\n')
        for ep in all_epochs:
            train_loss = round(float(np.mean(epoch_train_losses.get(ep, [0]))), 4)
            train_acc = round(float(
                train_eval_per_epoch.get(ep, {}).get('train_accuracy', 0.0)), 4)
            eval_loss = round(float(
                eval_per_epoch.get(ep, {}).get('eval_loss', 0.0)), 4)
            eval_acc = round(float(
                eval_per_epoch.get(ep, {}).get('eval_accuracy', 0.0)), 4)
            f.write(f'{ep};{train_loss};{train_acc};{eval_loss};{eval_acc}\n')

    print(f"\nTraining statistics saved to {stats_file}")


def finetune_model(train_texts, train_labels, val_texts, val_labels,
                   output_dir: str = OUTPUT_DIR, prompt_tuning: bool = False,
                   full_finetune: bool = False, resume_from: str = None,
                   iterations: int = None):
    """Finetune MamayLM using LoRA (default), prompt tuning, or full fine-tuning.

    When *iterations* is given, trains one epoch per iteration and saves
    weights after each to finetune-01/, finetune-02/, etc.  *resume_from*
    loads weights from a previous iteration folder so training can continue.

    Returns the path to the last saved model directory.
    """
    if full_finetune:
        mode_label = "FULL FINE-TUNING"
    elif prompt_tuning:
        mode_label = "PROMPT TUNING"
    else:
        mode_label = "LoRA"
    print("\n" + "="*80)
    print(f"STARTING FINETUNING ({mode_label})")
    print("="*80)

    # Load model and tokenizer
    tokenizer, model = load_base_model(MODEL_NAME)

    # Apply the chosen training method (with optional resume from checkpoint)
    if full_finetune:
        include_system_prompt = True
        lr = FULL_FINETUNE_LEARNING_RATE
        if resume_from:
            print(f"Resuming full fine-tune from: {resume_from}")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            tokenizer = AutoTokenizer.from_pretrained(resume_from)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                resume_from, device_map="cuda:0",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        print(f"Full fine-tuning: all {sum(p.numel() for p in model.parameters()):,} parameters trainable")
    elif prompt_tuning:
        include_system_prompt = False  # learned embeddings replace system prompt
        lr = PROMPT_TUNING_LEARNING_RATE
        if resume_from:
            print(f"Resuming prompt-tuning from: {resume_from}")
            model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
            nvt = model.peft_config["default"].num_virtual_tokens
            patch_peft_forward_for_gemma3(model, nvt)
        else:
            model, num_virtual_tokens = setup_prompt_tuning(model, tokenizer)
            patch_peft_forward_for_gemma3(model, num_virtual_tokens)
    else:
        include_system_prompt = True
        lr = LEARNING_RATE
        if resume_from:
            print(f"Resuming LoRA from: {resume_from}")
            model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
            model.print_trainable_parameters()
        else:
            model = setup_lora(model)

    # Determine suffix for the training stats CSV
    if full_finetune:
        suffix = "_full_finetuned"
    elif prompt_tuning:
        suffix = "_prompt_tuned"
    else:
        suffix = "_finetuned"

    # Prepare datasets
    train_dataset = prepare_dataset(train_texts, train_labels, tokenizer,
                                    include_system_prompt=include_system_prompt)
    val_dataset = prepare_dataset(val_texts, val_labels, tokenizer,
                                  include_system_prompt=include_system_prompt)

    if iterations is not None:
        # ── Iterative training: one epoch per iteration ──────────────
        # Determine starting iteration from resume folder name
        start_iter = 0
        if resume_from:
            try:
                start_iter = int(Path(resume_from).name.split('-')[-1])
            except (ValueError, IndexError):
                pass
            print(f"Continuing from iteration {start_iter}")

        last_output = None
        for i in range(1, iterations + 1):
            iter_num = start_iter + i
            iter_dir = f"finetune-{iter_num:02d}"
            Path(iter_dir).mkdir(exist_ok=True)

            print(f"\n{'─'*80}")
            print(f"ITERATION {i}/{iterations}  (cumulative epoch {iter_num})")
            print(f"Output: {iter_dir}")
            print(f"{'─'*80}")

            epoch_callback = _EpochTrainEvalCallback()

            training_args = TrainingArguments(
                output_dir=iter_dir,
                num_train_epochs=1,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                learning_rate=lr,
                warmup_steps=WARMUP_STEPS if (i == 1 and not resume_from) else 0,
                logging_steps=10,
                eval_strategy="epoch",
                save_strategy="no",
                bf16=True,
                optim="adamw_torch",
                remove_unused_columns=False,
                report_to="none",
                gradient_checkpointing=full_finetune,
                max_grad_norm=1.0,
                dataloader_pin_memory=False,
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=_compute_token_accuracy,
                preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
                callbacks=[epoch_callback],
            )
            epoch_callback.set_trainer(trainer)

            print(f"\nTraining iteration {i}...")
            trainer.train(ignore_keys_for_eval=["past_key_values"])

            # Save training statistics
            _save_training_stats(trainer, epoch_callback, suffix, iter_dir)

            # Save model
            print(f"Saving model to {iter_dir}...")
            trainer.save_model(iter_dir)
            tokenizer.save_pretrained(iter_dir)

            if prompt_tuning:
                pt_path = str(Path(iter_dir) / "prompt_embeddings.pt")
                extract_and_save_prompt_embeddings(trainer.model, pt_path)

            # Keep model for next iteration, free trainer
            model = trainer.model
            del trainer, epoch_callback
            torch.cuda.empty_cache()
            gc.collect()

            last_output = iter_dir
            print(f"Iteration {i}/{iterations} complete -> {iter_dir}")
    else:
        # ── Original single-run training ─────────────────────────────
        epoch_callback = _EpochTrainEvalCallback()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=lr,
            warmup_steps=WARMUP_STEPS,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=True,
            optim="adamw_torch",
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=full_finetune,
            max_grad_norm=1.0,
            dataloader_pin_memory=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=_compute_token_accuracy,
            preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
            callbacks=[epoch_callback],
        )
        epoch_callback.set_trainer(trainer)

        print("\nStarting training...")
        trainer.train(ignore_keys_for_eval=["past_key_values"])

        _save_training_stats(trainer, epoch_callback, suffix)

        print(f"\nSaving finetuned model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        if prompt_tuning:
            pt_path = str(Path(output_dir) / "prompt_embeddings.pt")
            extract_and_save_prompt_embeddings(trainer.model, pt_path)

        del trainer
        last_output = output_dir

    # Clean up training resources
    del model
    del train_dataset
    del val_dataset
    torch.cuda.empty_cache()
    gc.collect()

    print("Finetuning complete!")
    print("="*80)
    return last_output


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


def load_original_model():
    """Load the original base MamayLM model without any fine-tuning."""
    print(f"Loading original base model: {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Original base model loaded successfully!")
    return tokenizer, model


def load_full_finetuned_model(model_path: str = FULL_FINETUNE_OUTPUT_DIR):
    """Load a fully fine-tuned model (all parameters updated, no PEFT adapters)."""
    print(f"Loading fully fine-tuned model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print("Fully fine-tuned model loaded successfully!")
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


def evaluate_model(test_texts, test_labels, test_sheet_names,
                   model_path: str = OUTPUT_DIR,
                   show_all_queries: bool = False,
                   prompt_tuning: bool = False,
                   original: bool = False,
                   full_finetune: bool = False):
    """Evaluate a model on the test set.

    Four model sources are supported:
      - original=True:       base MamayLM without any fine-tuning
      - full_finetune=True:  fully fine-tuned model (all parameters)
      - prompt_tuning=True:  prompt-tuned PEFT adapter
      - default (LoRA):      fine-tuned LoRA adapter

    The per-sheet CSV report uses the same format as process.py:
    semicolon-separated, columns sheet_name;n;lp;ln;pp;pn;tp;fp;tn;fn;
    accuracy;precision;recall;f1, with a TOTAL row at the end.
    """
    print("\n" + "="*80)
    if original:
        mode_label = "ORIGINAL (BASE)"
    elif full_finetune:
        mode_label = "FULLY FINE-TUNED"
    elif prompt_tuning:
        mode_label = "PROMPT-TUNED"
    else:
        mode_label = "FINETUNED (LoRA)"
    print(f"EVALUATING {mode_label} MODEL")
    print("="*80)

    # Load the appropriate model
    if original:
        tokenizer, model = load_original_model()
    elif full_finetune:
        tokenizer, model = load_full_finetuned_model(model_path)
    elif prompt_tuning:
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

    # Per-sheet statistics (same structure as process.py)
    print("\n" + "="*80)
    print("PER-SHEET STATISTICS")
    print("="*80)

    # Preserve original sheet order from the test file
    seen = set()
    unique_sheets = []
    for s in test_sheet_names:
        if s not in seen:
            seen.add(s)
            unique_sheets.append(s)

    sheet_metrics = []
    for sheet_name in unique_sheets:
        mask = test_sheet_names == sheet_name

        if mask.sum() == 0:
            continue

        sheet_labels = test_labels[mask]
        sheet_preds = predictions[mask]

        n = int(mask.sum())
        lp = int((sheet_labels == 1).sum())   # labeled positives
        ln = int((sheet_labels == 0).sum())   # labeled negatives
        pp = int((sheet_preds == 1).sum())    # predicted positives
        pn = int((sheet_preds == 0).sum())    # predicted negatives

        cm_sheet = confusion_matrix(sheet_labels, sheet_preds, labels=[0, 1])
        tp = int(cm_sheet[1, 1])
        tn = int(cm_sheet[0, 0])
        fp = int(cm_sheet[0, 1])
        fn = int(cm_sheet[1, 0])

        acc = round((tp + tn) / n, 3) if n > 0 else 0.0
        prec = round(tp / pp, 3) if pp > 0 else 0.0
        rec = round(tp / lp, 3) if lp > 0 else 0.0
        f1_s = round(2 * prec * rec / (prec + rec), 3) if (prec + rec) > 0 else 0.0

        print(f"\n{sheet_name}:")
        print(f"  n={n}  lp={lp}  ln={ln}  pp={pp}  pn={pn}")
        print(f"  Accuracy: {acc}  Precision: {prec}  Recall: {rec}  F1: {f1_s}")
        print(f"  TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

        sheet_metrics.append({
            'sheet_name': sheet_name,
            'n': n, 'lp': lp, 'ln': ln, 'pp': pp, 'pn': pn,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1_s,
        })

    # Write semicolon-separated CSV identical to process.py
    if original:
        suffix = "_original"
    elif full_finetune:
        suffix = "_full_finetuned"
    elif prompt_tuning:
        suffix = "_prompt_tuned"
    else:
        suffix = "_finetuned"
    stats_file = f'mamaylm{suffix}_statistics.csv'

    with open(stats_file, 'w', encoding='utf-8') as f:
        header = "sheet_name;n;lp;ln;pp;pn;tp;fp;tn;fn;accuracy;precision;recall;f1\n"
        f.write(header)

        total = {'n': 0, 'lp': 0, 'ln': 0, 'pp': 0, 'pn': 0,
                 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

        for m in sheet_metrics:
            line = (
                f"{m['sheet_name']};{m['n']};{m['lp']};{m['ln']};"
                f"{m['pp']};{m['pn']};{m['tp']};{m['fp']};"
                f"{m['tn']};{m['fn']};{m['accuracy']};{m['precision']};{m['recall']};{m['f1']}\n"
            )
            f.write(line)
            for k in total:
                total[k] += m[k]

        # TOTAL row
        t_acc = round((total['tp'] + total['tn']) / total['n'], 3) if total['n'] > 0 else 0.0
        t_prec = round(total['tp'] / total['pp'], 3) if total['pp'] > 0 else 0.0
        t_rec = round(total['tp'] / total['lp'], 3) if total['lp'] > 0 else 0.0
        t_f1 = round(2 * t_prec * t_rec / (t_prec + t_rec), 3) if (t_prec + t_rec) > 0 else 0.0

        total_line = (
            f"TOTAL;{total['n']};{total['lp']};{total['ln']};"
            f"{total['pp']};{total['pn']};{total['tp']};{total['fp']};"
            f"{total['tn']};{total['fn']};{t_acc};{t_prec};{t_rec};{t_f1}\n"
        )
        f.write(total_line)

    print(f"\nPer-sheet statistics saved to {stats_file}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Accuracy:  {t_acc}")
    print(f"Total Precision: {t_prec}")
    print(f"Total Recall:    {t_rec}")
    print(f"Total F1:        {t_f1}")
    print("="*80)

    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Finetune MamayLM using half of the data with optional evaluation"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--lora',
        action='store_true',
        help='Fine-tune using LoRA: update low-rank adapters on attention/MLP weights'
    )
    mode_group.add_argument(
        '--prompt-tuning',
        action='store_true',
        help='Fine-tune using prompt tuning: optimise only the system-prompt '
             'embeddings while keeping all network weights frozen'
    )
    mode_group.add_argument(
        '--full-finetune',
        action='store_true',
        help='Full fine-tuning: update all model parameters (no PEFT adapters). '
             'Much higher memory requirements than LoRA or prompt tuning.'
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
        '--original',
        action='store_true',
        help='Evaluate the original base model (no fine-tuning). '
             'Use with --evaluate or --eval-only.'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to finetuned model (default: auto-selected based on mode)'
    )
    parser.add_argument(
        '--train-data',
        type=str,
        default='PETs_Ukr_Train.xlsx',
        help='Path to training data file (default: PETs_Ukr_Train.xlsx)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='PETs_Ukr_Test.xlsx',
        help='Path to test data file (default: PETs_Ukr_Test.xlsx)'
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
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of training iterations (1 epoch each). '
             'Saves weights after each iteration to finetune-01/, finetune-02/, etc.'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume training from a previous iteration folder (e.g. finetune-03). '
             'Iteration numbering continues from the given checkpoint.'
    )

    args = parser.parse_args()

    # If --resume-from is given without --iterations, default to 1 iteration
    if args.resume_from and args.iterations is None:
        args.iterations = 1

    # Validate --resume-from path exists
    if args.resume_from and not Path(args.resume_from).is_dir():
        parser.error(f"--resume-from path does not exist: {args.resume_from}")

    # Require a fine-tune mode unless evaluating the original base model
    if not any([args.lora, args.prompt_tuning, args.full_finetune]) and not args.original:
        parser.error("one of --lora, --prompt-tuning, --full-finetune is required "
                     "(or --original to evaluate the base model without fine-tuning)")

    # Resolve default model path based on mode
    if args.model_path is None:
        if args.prompt_tuning:
            args.model_path = PROMPT_TUNING_OUTPUT_DIR
        elif args.full_finetune:
            args.model_path = FULL_FINETUNE_OUTPUT_DIR
        else:
            args.model_path = OUTPUT_DIR

    # Single phrase prediction mode
    if args.predict:
        print("\n" + "="*80)
        print("SINGLE PHRASE PREDICTION")
        print("="*80)
        print(f"Text: {args.predict}")
        print(f"Model: {args.model_path}")
        if args.original:
            mode = "original base"
        elif args.full_finetune:
            mode = "full fine-tune"
        elif args.prompt_tuning:
            mode = "prompt tuning"
        else:
            mode = "LoRA"
        print(f"Mode: {mode}")
        print()

        # Load model
        if args.original:
            tokenizer, model = load_original_model()
        elif args.prompt_tuning:
            tokenizer, model = load_prompt_tuned_model(args.model_path)
        elif args.full_finetune:
            tokenizer, model = load_full_finetuned_model(args.model_path)
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

    # Finetune model (unless eval-only) — train/val data loaded only when needed
    if not args.eval_only:
        train_texts, train_labels, val_texts, val_labels = load_train_val_data(args.train_data)
        last_dir = finetune_model(train_texts, train_labels, val_texts, val_labels,
                       args.model_path, prompt_tuning=args.prompt_tuning,
                       full_finetune=args.full_finetune,
                       resume_from=args.resume_from,
                       iterations=args.iterations)
        # Point model_path to the last saved directory for subsequent evaluation
        if last_dir:
            args.model_path = last_dir
    else:
        print("Skipping finetuning (--eval-only mode)")

    # Evaluate model if requested or if eval-only — test data loaded only when needed
    if args.evaluate or args.eval_only:
        test_texts, test_labels, test_sheet_names = load_test_data(args.test_data)
        evaluate_model(test_texts, test_labels, test_sheet_names,
                       args.model_path, args.show_all_queries,
                       prompt_tuning=args.prompt_tuning,
                       original=args.original,
                       full_finetune=args.full_finetune)
    else:
        print("\nSkipping evaluation. Use --evaluate flag to evaluate the model.")
        print(f"To evaluate later, run: python {__file__} --eval-only")

    print("\nDone!")


if __name__ == '__main__':
    main()
