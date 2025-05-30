"""
Logging utilities for the Invoice Data Extraction project.
Provides centralized logging configuration with different log levels and formatters.
"""

import logging
import os
from datetime import datetime
from typing import Optional

class InvoiceLogger:
    """Custom logger class for invoice extraction project."""
    
    def __init__(self, name: str = "invoice_extraction", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with file and console handlers."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        if self.logger.handlers:
            return
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        log_filename = f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, log_filename),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, **kwargs)
    
    def log_processing_start(self, pdf_path: str):
        self.info(f"Starting processing for PDF: {os.path.basename(pdf_path)}")
    
    def log_processing_complete(self, pdf_path: str, processing_time: float):
        self.info(f"Completed processing {os.path.basename(pdf_path)} in {processing_time:.2f} seconds")
    
    def log_module_start(self, module_name: str):
        self.debug(f"Starting module: {module_name}")
    
    def log_module_complete(self, module_name: str, execution_time: float):
        self.debug(f"Completed module {module_name} in {execution_time:.2f} seconds")
    
    def log_field_extraction(self, field_name: str, confidence: float, value: str = None):
        if value:
            self.debug(f"Extracted {field_name}: '{value}' (confidence: {confidence:.2f})")
        else:
            self.warning(f"Failed to extract {field_name} (confidence: {confidence:.2f})")
    
    def log_verification_result(self, check_name: str, passed: bool, details: str = None):
        status = "PASSED" if passed else "FAILED"
        message = f"Verification {check_name}: {status}"
        if details:
            message += f" - {details}"
        if passed:
            self.debug(message)
        else:
            self.warning(message)
    
    def log_error_with_context(self, error: Exception, context: str):
        self.error(f"Error in {context}: {str(error)}", exc_info=True)

# Module-level functions for easy access
_default_logger = None

def get_logger(name: str = "invoice_extraction") -> InvoiceLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = InvoiceLogger(name)
    return _default_logger

def setup_logger(name: str = "invoice_extraction") -> InvoiceLogger:
    """Alias to maintain compatibility with older imports."""
    return get_logger(name)

def setup_module_logger(module_name: str) -> InvoiceLogger:
    return InvoiceLogger(f"invoice_extraction.{module_name}")

def log_function_call(func_name: str, **kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            logger.debug(f"Calling function: {func_name} with args: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func_name} failed: {str(e)}")
                raise
        return wrapper
    return decorator
