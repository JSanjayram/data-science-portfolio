import os
import json
from datetime import datetime
from typing import Any, Dict

def ensure_dir(directory: str):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)

def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict:
    """Load JSON file to dictionary"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().isoformat()

def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence:.1%}"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length"""
    return text[:max_length] + "..." if len(text) > max_length else text