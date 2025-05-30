"""
Utils package for Invoice Data Extraction Project
Contains utility modules for configuration, logging, and helper functions.
"""

from .config import Config, get_config
from .logger import setup_logger, get_logger
from .helpers import (
    validate_file_path,
    create_directory,
    get_file_extension,
    calculate_processing_time,
    format_currency,
    clean_text,
    extract_numbers,
    validate_gst_number,
    validate_date_format,
    calculate_confidence_score,
    merge_bounding_boxes,
    resize_image_if_needed,
    safe_divide,
    round_to_decimal_places
)

__version__ = "1.0.0"
__author__ = "Invoice Extraction Team"

# Package-level configuration
DEFAULT_CONFIG = {
    'ocr_engine': 'tesseract',
    'confidence_threshold': 0.7,
    'max_image_size': (3000, 3000),
    'supported_formats': ['.pdf', '.png', '.jpg', '.jpeg', '.tiff'],
    'output_formats': ['json', 'excel', 'csv']
}

# Export main classes and functions
__all__ = [
    'Config',
    'get_config',
    'setup_logger',
    'get_logger',
    'validate_file_path',
    'create_directory',
    'get_file_extension',
    'calculate_processing_time',
    'format_currency',
    'clean_text',
    'extract_numbers',
    'validate_gst_number',
    'validate_date_format',
    'calculate_confidence_score',
    'merge_bounding_boxes',
    'resize_image_if_needed',
    'safe_divide',
    'round_to_decimal_places',
    'DEFAULT_CONFIG'
]