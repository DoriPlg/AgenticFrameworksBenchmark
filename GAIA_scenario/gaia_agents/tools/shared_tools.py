import os
import io
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader
from ddgs import DDGS
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import contextlib

# --- 1. Web Search (Information Retrieval) ---
# Uses DuckDuckGo (No API Key required) to keep it equal for all frameworks.

class SearchInput(BaseModel):
    query: str = Field(description="The search query to find information.")

def web_search(query: str) -> str:
    """
    Search the web for general information, facts, or current events.
    Returns the top 5 snippets.
    """
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        formatted = "\n".join([f"- {r['title']}: {r['body']} (URL: {r['href']})" for r in results])
        return formatted
    except Exception as e:
        return f"Search failed: {str(e)}"

# --- 2. Web Browser (Reading specific pages) ---
# A simple scraper to "read" a page found via search.

class BrowserInput(BaseModel):
    url: str = Field(description="The URL of the webpage to read.")

# Blacklisted domains/patterns to prevent accessing answer keys
BLACKLISTED_PATTERNS = [
    'huggingface.co/datasets/gaia-benchmark',
    'huggingface.co/spaces/gaia-benchmark',
    'github.com/gaia-benchmark',
    'paperswithcode.com/dataset/gaia',
    # Add any leaderboard or answer discussion pages
    'reddit.com/r/MachineLearning.*gaia',
    'github.com.*gaia.*answer',
    'gaia.*validation.*json',
    'gaia.*test.*json',
]

def is_url_blacklisted(url: str) -> bool:
    """Check if URL matches any blacklisted pattern."""
    import re
    url_lower = url.lower()
    for pattern in BLACKLISTED_PATTERNS:
        if re.search(pattern.lower(), url_lower):
            return True
    return False

def read_webpage(url: str) -> str:
    """
    Visit a specific URL and extract its text content. 
    Use this after searching to get details.
    """
    # Check blacklist
    if is_url_blacklisted(url):
        return "ERROR: Access to this URL is blocked as it may contain validation data."
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        # ... rest of code
        # Remove script/style elements for cleanliness
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        return text[:8000] + "..." if len(text) > 8000 else text  # Truncate to fit context
    except Exception as e:
        return f"Could not read webpage: {str(e)}"

# --- 3. File Inspector (Multi-modality Handler) ---
# GAIA relies heavily on attached files (Excel, PDF, Text).

class FileToolInput(BaseModel):
    file_path: str = Field(description="The local path to the file you want to inspect.")
    query: Optional[str] = Field(description="For CSV/Excel, a pandas query. For PDF, a specific page number or keyword.", default=None)

def inspect_file(file_path: str, query: str = None) -> str:
    """
    Reads and analyzes local files (CSV, Excel, PDF, TXT, JSON, etc.).
    For CSV/Excel: Shows structure and preview. Use Python executor for complex queries.
    For PDF: Extracts all text content.
    For structured data: Always use Python executor for analysis after inspection.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."
    
    ext = file_path.split('.')[-1].lower()
    
    try:
        if ext in ['csv']:
            df = pd.read_csv(file_path)
            info = f"CSV File: {file_path}\n"
            info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            info += f"Columns: {list(df.columns)}\n"
            info += f"\nFirst 5 rows:\n{df.head(5).to_markdown()}\n"
            info += f"\nTip: Use Python executor with pd.read_csv('{file_path}') for analysis."
            return info
        
        elif ext in ['xlsx', 'xls']:
            xl_file = pd.ExcelFile(file_path)
            info = f"Excel File: {file_path}\n"
            info += f"Sheets: {xl_file.sheet_names}\n\n"
            df = pd.read_excel(file_path, sheet_name=0)
            info += f"First sheet shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            info += f"Columns: {list(df.columns)}\n"
            info += f"\nFirst 5 rows:\n{df.head(5).to_markdown()}\n"
            info += f"\nTip: Use Python executor with pd.read_excel('{file_path}', sheet_name='...') for analysis."
            return info
            
        elif ext == 'pdf':
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            text = ""
            for i, page in enumerate(reader.pages):
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text()
            
            # If too long, truncate but mention it
            if len(text) > 15000:
                text = text[:15000] + f"\n\n[Truncated. PDF has {num_pages} pages total. Use Python executor with pypdf for full access]"
            return f"PDF File: {file_path} ({num_pages} pages)\n{text}"
            
        elif ext in ['txt', 'md', 'py', 'json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) > 10000:
                content = content[:10000] + "\n\n[Content truncated. Use Python executor for full access]"
            return f"File: {file_path}\n\n{content}"
                
        else:
            return f"Unsupported file format: .{ext}\nUse Python Code Interpreter to handle this file."
            
    except Exception as e:
        return f"Error reading file: {str(e)}"

# --- 4. Python Code Interpreter (The "Super Tool") ---
# GAIA requires calculation and complex data processing.
# WARNING: Use `exec` cautiously. For a local benchmark, this is fine. 
# For production, use a sandboxed environment like E2B.

class PythonInput(BaseModel):
    code: str = Field(description="Valid Python code to execute. Use print() to output results.")

def python_interpreter(code: str) -> str:
    """
    Executes Python code to perform math, data analysis, or processing.
    Use this for ANY calculation, data manipulation, or complex logic.
    Available libraries: pandas (pd), numpy (np), requests, os, json, math, datetime, re.
    Always use print() to output your results.
    Existing variables are NOT preserved between calls (stateless).
    """
    import numpy as np
    import json
    import math
    import datetime
    import re
    
    # Create a buffer to capture stdout
    output_buffer = io.StringIO()
    
    try:
        # Redirect stdout to our buffer
        with contextlib.redirect_stdout(output_buffer):
            # Define execution environment with common libraries
            exec_globals = {
                "pd": pd,
                "np": np,
                "requests": requests,
                "os": os,
                "json": json,
                "math": math,
                "datetime": datetime,
                "re": re,
                "__builtins__": __builtins__,
            }
            exec(code, exec_globals)
            
        result = output_buffer.getvalue()
        if not result:
            return "Code executed successfully but printed no output. Did you forget print()?"
        return result.strip()
        
    except Exception as e:
        return f"Python Execution Error: {str(e)}\nMake sure to use print() for output."
    finally:
        output_buffer.close()

# --- Dictionary of Tools for Easy Import ---
GAIA_TOOLS = [web_search, read_webpage, inspect_file, python_interpreter]