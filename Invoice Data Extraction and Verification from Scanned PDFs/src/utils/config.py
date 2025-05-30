"""
Configuration management for Invoice Data Extraction Project
Handles all configuration settings, environment variables, and project parameters.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class OCRConfig:
    """OCR engine configuration settings."""
    engine: str = "tesseract"  # tesseract, easyocr
    language: str = "eng"
    confidence_threshold: float = 0.7
    page_segmentation_mode: int = 6  # Tesseract PSM
    dpi: int = 300
    timeout: int = 30  # seconds
    
    # Tesseract specific settings
    tesseract_config: str = "--oem 3 --psm 6"
    tesseract_path: Optional[str] = None
    
    # EasyOCR specific settings
    easyocr_gpu: bool = False
    easyocr_languages: list = field(default_factory=lambda: ['en'])


@dataclass
class ImageProcessingConfig:
    """Image preprocessing configuration."""
    max_image_size: tuple = (3000, 3000)
    min_image_size: tuple = (800, 600)
    auto_rotate: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    binarize: bool = True
    deskew: bool = True
    
    # Preprocessing parameters
    denoise_strength: int = 3
    contrast_factor: float = 1.2
    gaussian_blur_kernel: int = 3
    morphological_kernel: int = 2
    deskew_angle_threshold: float = 0.5


@dataclass
class ExtractionConfig:
    """Field extraction configuration."""
    confidence_threshold: float = 0.7
    date_formats: list = field(default_factory=lambda: [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y",
        "%B %d, %Y", "%d %B %Y", "%d-%b-%Y"
    ])
    
    # GST number patterns (India specific)
    gst_pattern: str = r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}'
    
    # Invoice number patterns
    invoice_patterns: list = field(default_factory=lambda: [
        r'INV[-/]?\d+',
        r'INVOICE[-/]?\d+',
        r'Bill[-/]?\d+',
        r'\d{4,}'
    ])
    
    # PO number patterns  
    po_patterns: list = field(default_factory=lambda: [
        r'PO[-/]?\d+',
        r'P\.O\.[-/]?\d+',
        r'Purchase[-/]?Order[-/]?\d+'
    ])
    
    # Currency symbols and formats
    currency_symbols: list = field(default_factory=lambda: ['₹', '$', '€', '£'])
    amount_patterns: list = field(default_factory=lambda: [
        r'₹\s*[\d,]+\.?\d*',
        r'\$\s*[\d,]+\.?\d*',
        r'[\d,]+\.?\d*'
    ])


@dataclass
class TableExtractionConfig:
    """Table parsing configuration."""
    min_table_rows: int = 2
    min_table_cols: int = 3
    table_detection_threshold: float = 0.6
    
    # Column detection parameters
    column_separator_threshold: int = 20
    row_separator_threshold: int = 10
    
    # Expected table headers (flexible matching)
    expected_headers: list = field(default_factory=lambda: [
        'description', 'qty', 'quantity', 'rate', 'price', 'amount', 
        'total', 'hsn', 'sac', 'gst', 'tax'
    ])


@dataclass
class SealDetectionConfig:
    """Seal and signature detection configuration."""
    enable_detection: bool = True
    min_seal_size: tuple = (50, 50)
    max_seal_size: tuple = (500, 500)
    
    # Detection parameters
    contour_area_threshold: int = 1000
    aspect_ratio_range: tuple = (0.5, 2.0)
    circularity_threshold: float = 0.3
    
    # Output settings
    save_cropped_seals: bool = True
    seal_image_format: str = "PNG"
    seal_background_removal: bool = True


@dataclass
class OutputConfig:
    """Output generation configuration."""
    output_directory: str = "output"
    create_json: bool = True
    create_excel: bool = True
    create_verification_report: bool = True
    
    # File naming
    json_filename: str = "extracted_data.json"
    excel_filename: str = "extracted_data.xlsx"
    verification_filename: str = "verifiability_report.json"
    
    # Excel formatting
    excel_auto_format: bool = True
    excel_include_charts: bool = False
    
    # JSON formatting
    json_indent: int = 2
    json_ensure_ascii: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    log_file: str = "logs/invoice_extraction.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_logging: bool = True


class Config:
    """Main configuration class that combines all config sections."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        # Initialize with default configurations
        self.ocr = OCRConfig()
        self.image_processing = ImageProcessingConfig()
        self.extraction = ExtractionConfig()
        self.table_extraction = TableExtractionConfig()
        self.seal_detection = SealDetectionConfig()
        self.output = OutputConfig()
        self.logging = LoggingConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
        
        # Set up derived configurations
        self._setup_derived_config()
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configurations
            self._update_from_dict(config_data)
            
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {str(e)}")
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'INVOICE_OCR_ENGINE': ('ocr', 'engine'),
            'INVOICE_OCR_LANGUAGE': ('ocr', 'language'),
            'INVOICE_CONFIDENCE_THRESHOLD': ('extraction', 'confidence_threshold'),
            'INVOICE_OUTPUT_DIR': ('output', 'output_directory'),
            'INVOICE_LOG_LEVEL': ('logging', 'level'),
            'INVOICE_TESSERACT_PATH': ('ocr', 'tesseract_path'),
            'INVOICE_MAX_IMAGE_WIDTH': ('image_processing', 'max_image_size', 0),
            'INVOICE_MAX_IMAGE_HEIGHT': ('image_processing', 'max_image_size', 1),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config_path, env_value)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _set_nested_value(self, config_path: tuple, value: str):
        """Set a nested configuration value."""
        try:
            obj = self
            for part in config_path[:-1]:
                obj = getattr(obj, part)
            
            # Convert value to appropriate type
            current_value = getattr(obj, config_path[-1])
            if isinstance(current_value, bool):
                value = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            elif isinstance(current_value, tuple) and len(config_path) > 2:
                # Handle tuple elements (like max_image_size)
                current_tuple = list(getattr(obj, config_path[-2]))
                current_tuple[config_path[-1]] = int(value)
                setattr(obj, config_path[-2], tuple(current_tuple))
                return
            
            setattr(obj, config_path[-1], value)
            
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Warning: Could not set config value {config_path}: {e}")
    
    def _setup_derived_config(self):
        """Set up derived configuration values."""
        # Ensure output directory exists
        os.makedirs(self.output.output_directory, exist_ok=True)
        
        # Ensure log directory exists
        log_dir = Path(self.logging.log_file).parent
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up full file paths
        self.output.json_path = os.path.join(
            self.output.output_directory, 
            self.output.json_filename
        )
        self.output.excel_path = os.path.join(
            self.output.output_directory, 
            self.output.excel_filename
        )
        self.output.verification_path = os.path.join(
            self.output.output_directory, 
            self.output.verification_filename
        )
        self.output.seal_directory = os.path.join(
            self.output.output_directory, 
            "seal_signatures"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'ocr': self.ocr.__dict__,
            'image_processing': self.image_processing.__dict__,
            'extraction': self.extraction.__dict__,
            'table_extraction': self.table_extraction.__dict__,
            'seal_detection': self.seal_detection.__dict__,
            'output': self.output.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_to_file(self, config_file: str):
        """Save current configuration to file."""
        config_path = Path(config_file)
        config_dict = self.to_dict()
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"Error saving configuration file: {str(e)}")
    
    def validate(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate OCR settings
        if self.ocr.engine not in ['tesseract', 'easyocr']:
            issues.append("OCR engine must be 'tesseract' or 'easyocr'")
        
        if not 0 <= self.ocr.confidence_threshold <= 1:
            issues.append("OCR confidence threshold must be between 0 and 1")
        
        # Validate image processing settings
        if any(size <= 0 for size in self.image_processing.max_image_size):
            issues.append("Max image size must be positive")
        
        # Validate extraction settings
        if not 0 <= self.extraction.confidence_threshold <= 1:
            issues.append("Extraction confidence threshold must be between 0 and 1")
        
        # Validate output settings
        if not os.path.exists(os.path.dirname(self.output.output_directory)):
            parent_dir = os.path.dirname(self.output.output_directory)
            if parent_dir and not os.path.exists(parent_dir):
                issues.append(f"Output directory parent does not exist: {parent_dir}")
        
        return issues
    
    def get_summary(self) -> str:
        """Get a human-readable summary of current configuration."""
        summary = []
        summary.append("=== Invoice Extraction Configuration ===")
        summary.append(f"OCR Engine: {self.ocr.engine}")
        summary.append(f"OCR Language: {self.ocr.language}")
        summary.append(f"Confidence Threshold: {self.extraction.confidence_threshold}")
        summary.append(f"Max Image Size: {self.image_processing.max_image_size}")
        summary.append(f"Output Directory: {self.output.output_directory}")
        summary.append(f"Log Level: {self.logging.level}")
        summary.append(f"Seal Detection: {'Enabled' if self.seal_detection.enable_detection else 'Disabled'}")
        
        return "\n".join(summary)


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_file: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_file: Path to configuration file
        reload: Force reload of configuration
        
    Returns:
        Global configuration instance
    """
    global _global_config
    
    if _global_config is None or reload:
        _global_config = Config(config_file)
    
    return _global_config


def create_default_config_file(output_path: str = "config.json"):
    """Create a default configuration file."""
    default_config = Config()
    default_config.save_to_file(output_path)
    return output_path


# Example usage and testing
if __name__ == "__main__":
    # Create and test configuration
    config = Config()
    
    print("Default Configuration:")
    print(config.get_summary())
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid.")
    
    # Save default configuration
    config.save_to_file("default_config.json")
    print("\nDefault configuration saved to 'default_config.json'")