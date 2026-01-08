"""Unified LLM configuration for all frameworks.

This module provides LLM configuration and client that works across
PydanticAI, LangGraph, CrewAI, and Smolagents with support for different models.
"""

import os
import json
from typing import Dict, List, Union, Any, Optional
from dotenv import load_dotenv
import requests
# from openai import OpenAI


def get_available_models(base_url: str = None) -> List[str]:
    """Fetch available models from the LLM proxy endpoint.
    
    Args:
        base_url: Optional base URL for the models endpoint.
                 If not provided, uses LLM_PROXY_PORT from environment.
    
    Returns:
        List of available model names
    """
    if base_url is None:
        port = os.getenv("LLM_PROXY_PORT", "54844")
        base_url = f"http://host.docker.internal:{port}"
    
    try:
        response = requests.get(f"{base_url}/llm/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        return [model["model_name"] for model in data]
    except Exception as e:
        print(f"Warning: Could not fetch models from {base_url}/llm/models: {e}")
        return []

models= get_available_models()
# Load environment variables
load_dotenv()


def normalize_messages(messages) -> List[Dict[str, str]]:
    """Normalize message content to ensure compatibility with Mistral.
    
    Handles both dictionary and object message formats.
    """
    normalized = []
    for msg in messages:
        # Handle both dictionary and object access
        if hasattr(msg, 'get'):  # Dictionary-like
            content = msg.get('content', '')
            role = msg.get('role', 'user')
            name = msg.get('name')
        else:  # Object with attributes
            content = getattr(msg, 'content', '')
            role = getattr(msg, 'role', 'user')
            name = getattr(msg, 'name', None)
        
        # Ensure content is a string and strip any problematic characters
        if not isinstance(content, str):
            content = str(content)
        
        # Create a clean message copy
        clean_msg = {
            'role': role,
            'content': content.strip()
        }
        
        # Only include name if it exists and is a string
        if name and isinstance(name, str):
            clean_msg['name'] = name.strip()
            
        normalized.append(clean_msg)
    return normalized


def get_llm_config(model_choice: int = 0) -> dict:
    """
    Get LLM configuration as a dictionary.
    
    Args:
        model_choice (int): Index of the model to use (default: 0)
            0: Meta-Llama-3_3-70B-Instruct
            1: Qwen2.5-Coder-32B-Instruct
            2: DeepSeek-R1-Distill-Llama-70B
            3: Mistral-Nemo-Instruct-2407
            4: gpt-oss-20b
            5: Qwen2.5-VL-72B-Instruct
            6: Qwen3-32B
            7: Llama-3.1-8B-Instruct
            8: Mistral-Small-3.2-24B-Instruct-2506
            9: Mixtral-8x7B-Instruct-v0.1
            10: gpt-oss-120b
            11: Mistral-7B-Instruct-v0.3
    
    Returns:
        dict: Configuration with model, base_url, and api_key
    """
    if model_choice >= len(models):
        raise ValueError(f"Invalid model choice: {model_choice}")
    model = models[model_choice]
    
    # Special handling for Mistral
    if 'mistral' in model.lower():
        return {
            'model': model,
            'base_url': os.getenv('OPENAI_API_BASE', f'http://host.docker.internal:{os.getenv("LLM_PROXY_PORT", "54844")}/ai-gen-proxy/llm/ovh/v1'),
            'api_key': os.getenv('OPENAI_API_KEY', 'dummy-key-not-needed'),
            'http_client': None,  # Will be set by the framework
            'message_normalizer': normalize_messages
        }
    
    # Default configuration for other models
    return {
        'model': model,
        'base_url': os.getenv('OPENAI_API_BASE', 
        f'http://host.docker.internal:{os.getenv("LLM_PROXY_PORT", "54844")}/ai-gen-proxy/llm/ovh/v1'),
        'api_key': os.getenv('OPENAI_API_KEY', 'dummy-key-not-needed'),
    }

# class OvhClient(OpenAI):
#     def __init__(self, model_num=0, **kwargs):
#         # Configure to use Ollama's endpoint
#         kwargs["base_url"] = os.getenv('OPENAI_API_BASE', f'http://host.docker.internal:{os.getenv("LLM_PROXY_PORT", "54844")}/ai-gen-proxy/llm/ovh/v1')

#         # Ollama doesn't require an API key but the client expects one
#         kwargs["api_key"] = "dummy-key-not-needed"

#         super().__init__(**kwargs)
#         self.model_name = models[model_num]
        
#         # Check if the model exists
#         print(f"Using Ovh model: {self.model_name}")

#     def create_completion(self, *args, **kwargs):
#         # Override model name if not explicitly provided
#         if "model" not in kwargs:
#             kwargs["model"] = self.model_name

#         return super().create_completion(*args, **kwargs)

#     def create_chat_completion(self, *args, **kwargs):
#         # Override model name if not explicitly provided
#         if "model" not in kwargs:
#             kwargs["model"] = self.model_name

#         return super().create_chat_completion(*args, **kwargs)
        
#     # These methods are needed for compatibility with agents library
#     def completion(self, prompt, **kwargs):
#         if "model" not in kwargs:
#             kwargs["model"] = self.model_name
#         return self.completions.create(prompt=prompt, **kwargs)
        
#     def chat_completion(self, messages, **kwargs):
#         if "model" not in kwargs:
#             kwargs["model"] = self.model_name
#         return self.chat.completions.create(messages=messages, **kwargs)
    