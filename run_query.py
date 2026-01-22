#!/usr/bin/env python3
"""
Run MamayLM with a specific query
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    print("Loading MamayLM model...")
    model_name = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print("Model loaded successfully!\n")
    
    # User's query
    query = "Серед поданих нижче слів обери слово, яке написане в кутових дужках і напиши його без кутових дужок: \nЯблуко <майстер> груша куля"
    print(f"Query: {query}\n")
    
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse:\n{response}\n")

if __name__ == "__main__":
    main()
