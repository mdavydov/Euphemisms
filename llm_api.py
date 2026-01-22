"""LLM API interface for different models."""

import os
import json
import time
import requests
import google.generativeai as genai
from openai import OpenAI
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import SYSTEM_PROMPT, SUPPORTED_MODELS


class LLMClient:
    """Base class for LLM API clients."""
    
    def __init__(self, model_name: str, api_key: str, queries_per_minute: Optional[int] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.queries_per_minute = queries_per_minute
        
        # Calculate delay between requests based on rate limit
        if queries_per_minute:
            self.rate_limit_delay = 60.0 / queries_per_minute
        else:
            self.rate_limit_delay = 0.1  # Default delay between requests
        
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def process_text(self, text: str) -> str:
        """Process a single text with the LLM. To be implemented by subclasses."""
        raise NotImplementedError
    
    def process_batch(self, texts: List[str], max_workers: int = 10) -> List[str]:
        """Process multiple texts in parallel."""
        results = []
        
        # If rate limiting is strict, process sequentially
        if self.queries_per_minute and self.queries_per_minute < 10:
            max_workers = 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_text = {
                executor.submit(self.process_text, text): text 
                for text in texts
            }
            
            # Collect results in order
            for future in as_completed(future_to_text):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing text: {e}")
                    results.append("0 Error")
        
        return results


class DeepSeekClient(LLMClient):
    """DeepSeek API client."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", queries_per_minute: Optional[int] = None):
        super().__init__(model, api_key, queries_per_minute)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    def process_text(self, text: str) -> str:
        """Process text using DeepSeek API."""
        self._wait_for_rate_limit()
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                store=True
            )
            
            result = completion.choices[0].message.content
            print(f"DeepSeek result: {result[:100]}...")
            return result
            
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return "0 Error"


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", queries_per_minute: Optional[int] = None):
        super().__init__(model, api_key, queries_per_minute)
        self.client = OpenAI(api_key=api_key)
    
    def process_text(self, text: str) -> str:
        """Process text using OpenAI API."""
        self._wait_for_rate_limit()
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ]
            )
            
            result = completion.choices[0].message.content
            print(f"OpenAI result: {result[:100]}...")
            return result
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "0 Error"


class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", queries_per_minute: Optional[int] = None):
        super().__init__(model, api_key, queries_per_minute)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        # Override default delay if not explicitly set via queries_per_minute
        if not queries_per_minute:
            self.rate_limit_delay = 1.0  # Gemini has stricter rate limits
    
    def process_text(self, text: str) -> str:
        """Process text using Gemini API."""
        self._wait_for_rate_limit()
        try:
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}"
            response = self.model.generate_content(prompt)
            
            result = response.text
            print(f"Gemini result: {result[:100]}...")
            return result
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "0 Error"



class MamayLMClient(LLMClient):
    """MamayLM local inference client using Hugging Face transformers."""
    
    def __init__(self, model_name: str = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"):
        super().__init__(model_name, api_key=None)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "MamayLM requires transformers and torch. "
                "Please install them with: uv pip install transformers torch"
            )
        
        print(f"Loading MamayLM model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("MamayLM model loaded successfully!")
        self.rate_limit_delay = 0  # No rate limit for local inference

    def process_text(self, text: str) -> str:
        """Process text using local MamayLM model."""
        try:
            import torch
            
            # Format the prompt with system prompt
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {text}"
            print(prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the result
            result = result[len(prompt):].strip()
            
            # Clean up GPU memory
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            print(f"MamayLM result: {result[:100]}...")
            return result
            
        except Exception as e:
            print(f"MamayLM inference error: {e}")
            # Try to clean up on error too
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "0 Error"

def create_llm_client(provider: str, api_key: str, model: Optional[str] = None, queries_per_minute: Optional[int] = None) -> LLMClient:
    """
    Factory function to create the appropriate LLM client.
    
    Args:
        provider: The LLM provider ('deepseek', 'openai', 'gemini')
        api_key: API key for the provider
        model: Specific model name (optional, uses default if not provided)
        queries_per_minute: Maximum number of API queries per minute (optional)
    
    Returns:
        LLMClient instance
    """
    if provider not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {list(SUPPORTED_MODELS.keys())}")
    
    model_config = SUPPORTED_MODELS[provider]
    model_name = model or model_config['default_model']
    
    if provider == 'deepseek':
        return DeepSeekClient(api_key, model_name, queries_per_minute)
    elif provider == 'openai':
        return OpenAIClient(api_key, model_name, queries_per_minute)
    elif provider == 'gemini':
        return GeminiClient(api_key, model_name, queries_per_minute)
    elif provider == 'mamaylm':
        return MamayLMClient(model_name)
    else:
        raise ValueError(f"Provider {provider} not implemented")


def get_api_key(provider: str) -> str:
    """Get API key from environment variables."""
    if provider not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # MamayLM is local, doesn't need an API key
    if provider == 'mamaylm':
        return None
    
    env_var = SUPPORTED_MODELS[provider]['api_key_env']
    api_key = os.getenv(env_var)
    
    if not api_key:
        raise ValueError(f"API key not found in environment variable {env_var}")
    
    return api_key


def clean_strings(strings: List[str]) -> List[str]:
    """Clean text strings (extracted from notebooks)."""
    return [str(s).strip('\n') for s in strings]


def extract_label_from_response(response: str) -> int:
    """
    Extract label (0 or 1) from LLM response.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Integer label (0 or 1)
    """
    if not response or response == "0 Error":
        return 0
    
    # Look for the first digit in the response
    for char in response:
        if char == '1':
            return 1
        elif char == '0':
            return 0
    
    # Default to 0 if no clear label found
    return 0
