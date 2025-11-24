"""Unified LLM configuration for all frameworks.

This module provides LLM configuration and client that works across
PydanticAI, LangGraph, CrewAI, and Smolagents with support for different models.
"""

import os
import json
from typing import Dict, List, Union, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

models=(
    "Meta-Llama-3_3-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "Mixtral-8x7B-Instruct-v0.1",
    "gpt-oss-120b",
    "gpt-oss-20b"
)
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
            1: Llama-3.1-8B-Instruct
            2: Mistral-Small-3.2-24B-Instruct-2506
            3: Mixtral-8x7B-Instruct-v0.1
            4: gpt-oss-120b
            5: gpt-oss-20b
    
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
            'base_url': os.getenv('OPENAI_API_BASE', f'http://localhost:{os.getenv("LLM_PROXY_PORT", "54844")}/ai-gen-proxy/llm/ovh/v1'),
            'api_key': os.getenv('OPENAI_API_KEY', 'dummy-key-not-needed'),
            'http_client': None,  # Will be set by the framework
            'message_normalizer': normalize_messages
        }
    
    # Default configuration for other models
    return {
        'model': model,
        'base_url': os.getenv('OPENAI_API_BASE', 
        f'http://localhost:{os.getenv("LLM_PROXY_PORT", "54844")}/ai-gen-proxy/llm/ovh/v1'),
        'api_key': os.getenv('OPENAI_API_KEY', 'dummy-key-not-needed'),
    }

class OvhClient(OpenAI):
    def __init__(self, model_num=0, **kwargs):
        # Configure to use Ollama's endpoint
        kwargs["base_url"] = os.getenv('OPENAI_API_BASE', f'http://localhost:{os.getenv("LLM_PROXY_PORT", "54844")}/ai-gen-proxy/llm/ovh/v1')

        # Ollama doesn't require an API key but the client expects one
        kwargs["api_key"] = "dummy-key-not-needed"

        super().__init__(**kwargs)
        self.model_name = models[model_num]
        
        # Check if the model exists
        print(f"Using Ovh model: {self.model_name}")

    def create_completion(self, *args, **kwargs):
        # Override model name if not explicitly provided
        if "model" not in kwargs:
            kwargs["model"] = self.model_name

        return super().create_completion(*args, **kwargs)

    def create_chat_completion(self, *args, **kwargs):
        # Override model name if not explicitly provided
        if "model" not in kwargs:
            kwargs["model"] = self.model_name

        return super().create_chat_completion(*args, **kwargs)
        
    # These methods are needed for compatibility with agents library
    def completion(self, prompt, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self.model_name
        return self.completions.create(prompt=prompt, **kwargs)
        
    def chat_completion(self, messages, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self.model_name
        return self.chat.completions.create(messages=messages, **kwargs)