"""
Helper functions for the Invoice Data Extraction project.
Contains common utilities used across different modules.
"""

import os
import re
import json
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

# File and Directory Utilities

def validate_file_path(path: str) -> bool:
    """Validate if file path exists and is a file."""
    return isinstance(path, str) and os.path.isfile(path)

from pathlib import Path

def ensure_directory_exists(directory_path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    return directory_path

def get_temp_directory() -> str:
    """Create and return a temporary directory for processing."""
    return tempfile.mkdtemp(prefix="invoice_extraction_")

def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory and its contents."""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase."""
    return os.path.splitext(file_path)[1].lower()

def is_pdf_file(file_path: str) -> bool:
    """Check if file is a PDF."""
    return get_file_extension(file_path) == '.pdf'

def generate_output_filename(input_path: str, suffix: str, extension: str) -> str:
    """Generate output filename based on input file."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return f"{base_name}_{suffix}.{extension}"

# Text Processing Utilities

def clean_text(text: Optional[str]) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.\-\(\)\[\]\/\\:;,@#]', '', text)
    return text

def extract_numbers_from_text(text: Optional[str]) -> List[float]:
    """Extract all numeric values from text."""
    if not text:
        return []
    number_pattern = r'\d+\.?\d*'
    matches = re.findall(number_pattern, text)
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers

def extract_dates_from_text(text: Optional[str]) -> List[str]:
    """Extract dates from text in various formats."""
    if not text:
        return []
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD
        r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Mon YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',  # Mon DD, YYYY
    ]
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                # If regex contains groups, flatten
                matches = [' '.join(m) if isinstance(m, tuple) else m for m in matches]
            dates.extend(matches)
    return dates

def normalize_gst_number(gst_text: Optional[str]) -> str:
    """Normalize GST number format."""
    if not gst_text:
        return ""
    gst = re.sub(r'\s+', '', gst_text.upper())
    gst_pattern = r'^[0-9]{2}[A-Z0-9]{10}[A-Z][0-9][A-Z][0-9A-Z]$'
    if re.match(gst_pattern, gst):
        return gst
    return gst_text  # Return original if doesn't match pattern

# Mathematical Utilities

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def calculate_percentage(part: float, whole: float) -> float:
    """Calculate percentage with error handling."""
    return safe_divide(part * 100, whole, 0.0)

def round_to_currency(amount: float, decimals: int = 2) -> float:
    """Round amount to currency precision."""
    try:
        return round(float(amount), decimals)
    except (TypeError, ValueError):
        return 0.0

def validate_calculation(expected: float, actual: float, tolerance: float = 0.01) -> bool:
    """Validate if two values are equal within tolerance."""
    try:
        return abs(float(expected) - float(actual)) <= tolerance
    except (TypeError, ValueError):
        return False

# Confidence and Scoring Utilities

def calculate_weighted_confidence(confidences: List[float], weights: Optional[List[float]] = None) -> float:
    """Calculate weighted average confidence score."""
    if not confidences:
        return 0.0
    if weights is None:
        weights = [1.0] * len(confidences)
    if len(confidences) != len(weights):
        return sum(confidences) / len(confidences)  # Simple average fallback
    weighted_sum = sum(c * w for c, w in zip(confidences, weights))
    weight_sum = sum(weights)
    return safe_divide(weighted_sum, weight_sum, 0.0)

def normalize_confidence(confidence: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Normalize confidence score to 0-1 range."""
    try:
        normalized = (confidence - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0

def confidence_to_category(confidence: float) -> str:
    """Convert confidence score to category."""
    if confidence >= 0.9:
        return "HIGH"
    elif confidence >= 0.7:
        return "MEDIUM"
    elif confidence >= 0.5:
        return "LOW"
    else:
        return "VERY_LOW"

# Data Structure Utilities

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def safe_get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary value."""
    try:
        current = data
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default

def filter_empty_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty values from dictionary."""
    return {k: v for k, v in data.items() if v not in [None, "", [], {}]}

# Validation Utilities

def is_valid_email(email: Optional[str]) -> bool:
    """Check if email format is valid."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_valid_phone(phone: Optional[str]) -> bool:
    """Check if phone number format is valid (Indian format)."""
    if not phone:
        return False
    clean_phone = re.sub(r'[^\d+]', '', phone)
    patterns = [
        r'^\+91\d{10}$',
        r'^91\d{10}$',
        r'^\d{10}$',
    ]
    return any(re.match(pattern, clean_phone) for pattern in patterns)

def is_valid_pincode(pincode: Optional[str]) -> bool:
    """Check if Indian pincode format is valid."""
    if not pincode:
        return False
    return bool(re.match(r'^\d{6}$', pincode.strip()))

# Timing and Performance Utilities

def time_function(func):
    """Decorator to measure function execution time."""
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()

# Image Processing Utilities

def calculate_image_quality_score(image_array: np.ndarray) -> float:
    """Calculate a basic image quality score."""
    try:
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2)
        else:
            gray = image_array
        from scipy import ndimage
        laplacian_var = np.var(ndimage.laplace(gray))
        return min(1.0, laplacian_var / 1000.0)
    except Exception:
        return 0.5

# Error Handling Utilities

def safe_execute(func, *args, default=None, **kwargs):
    """Safely execute a function and return default on error."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default

