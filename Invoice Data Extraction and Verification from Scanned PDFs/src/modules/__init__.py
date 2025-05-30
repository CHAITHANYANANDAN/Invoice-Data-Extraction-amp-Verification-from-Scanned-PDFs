# ==========================================
# src/modules/__init__.py
# ==========================================
"""
Invoice extraction modules package
"""

# ==========================================
# src/modules/pdf_processor.py
# ==========================================
"""PDF to Image conversion module"""

import os
from typing import List
import logging
from pdf2image import convert_from_path
import numpy as np

class PDFProcessor:
    """Handles PDF to image conversion"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_images_from_pdf(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """Convert PDF pages to image arrays"""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=dpi)
            
            # Convert PIL images to numpy arrays
            image_arrays = []
            for img in images:
                img_array = np.array(img)
                image_arrays.append(img_array)
                
            self.logger.info(f"Converted {len(image_arrays)} pages from PDF")
            return image_arrays
            
        except Exception as e:
            self.logger.error(f"Error converting PDF: {e}")
            raise


# ==========================================
# src/modules/image_preprocessor.py
# ==========================================
"""Image preprocessing and enhancement module"""

import cv2
import numpy as np
from typing import Optional
import logging

class ImagePreprocessor:
    """Handles image preprocessing for better OCR results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply complete enhancement pipeline"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply enhancements
            denoised = self.denoise_image(gray)
            enhanced = self.enhance_contrast(denoised)
            binary = self.binarize_image(enhanced)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {e}")
            return image
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        return cv2.fastNlMeansDenoising(image)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Improve image contrast"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def binarize_image(self, image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """Convert to binary image"""
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary


# ==========================================
# src/modules/ocr_engine.py
# ==========================================
"""OCR text extraction module"""

import pytesseract
import easyocr
import numpy as np
from typing import Dict, List, Any
import logging

class OCREngine:
    """Handles OCR text extraction with multiple engines"""
    
    def __init__(self, engine_type: str = 'tesseract'):
        self.engine_type = engine_type
        self.logger = logging.getLogger(__name__)
        
        if engine_type == 'easyocr':
            self.reader = easyocr.Reader(['en'])
    
    def extract_text_with_coordinates(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text with bounding box coordinates"""
        try:
            if self.engine_type == 'tesseract':
                return self._extract_with_tesseract(image)
            else:
                return self._extract_with_easyocr(image)
                
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return {'text': '', 'words': [], 'confidence': 0.0}
    
    def _extract_with_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract using Tesseract"""
        # Get detailed data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extract text and coordinates
        words = []
        full_text = ""
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Filter low confidence
                word_info = {
                    'text': data['text'][i],
                    'bbox': (data['left'][i], data['top'][i], 
                            data['width'][i], data['height'][i]),
                    'confidence': float(data['conf'][i]) / 100.0
                }
                words.append(word_info)
                full_text += data['text'][i] + " "
        
        avg_confidence = sum(w['confidence'] for w in words) / len(words) if words else 0
        
        return {
            'text': full_text.strip(),
            'words': words,
            'confidence': avg_confidence
        }
    
    def _extract_with_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract using EasyOCR"""
        results = self.reader.readtext(image)
        
        words = []
        full_text = ""
        
        for (bbox, text, confidence) in results:
            word_info = {
                'text': text,
                'bbox': bbox,
                'confidence': confidence
            }
            words.append(word_info)
            full_text += text + " "
        
        avg_confidence = sum(w['confidence'] for w in words) / len(words) if words else 0
        
        return {
            'text': full_text.strip(),
            'words': words,
            'confidence': avg_confidence
        }


# ==========================================
# src/modules/field_extractor.py
# ==========================================
"""Structured field extraction module"""

import re
from typing import Dict, List, Any, Optional
import logging

class FieldExtractor:
    """Extracts structured fields from OCR text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define regex patterns
        self.patterns = {
            'invoice_number': [
                r'invoice\s*(?:no\.?|number)\s*:?\s*([A-Z0-9\-/]+)',
                r'inv\s*(?:no\.?|#)\s*:?\s*([A-Z0-9\-/]+)',
                r'bill\s*(?:no\.?|number)\s*:?\s*([A-Z0-9\-/]+)'
            ],
            'invoice_date': [
                r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'invoice\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            ],
            'gst_number': [
                r'gstin?\s*:?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9]{1}[Z]{1}[0-9A-Z]{1})',
                r'gst\s*(?:no\.?|number)\s*:?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9]{1}[Z]{1}[0-9A-Z]{1})'
            ],
            'po_number': [
                r'po\s*(?:no\.?|number)\s*:?\s*([A-Z0-9\-/]+)',
                r'purchase\s*order\s*:?\s*([A-Z0-9\-/]+)'
            ]
        }
    
    def extract_all_fields(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract all fields from OCR results"""
        # Combine text from all pages
        combined_text = ""
        for result in ocr_results:
            combined_text += result.get('text', '') + " "
        
        extracted = {}
        
        # Extract each field
        extracted['invoice_number'] = self.extract_invoice_number(combined_text)
        extracted['invoice_date'] = self.extract_invoice_date(combined_text)
        extracted['supplier_gst_number'] = self.extract_gst_number(combined_text, 'supplier')
        extracted['bill_to_gst_number'] = self.extract_gst_number(combined_text, 'bill_to')
        extracted['po_number'] = self.extract_po_number(combined_text)
        extracted['shipping_address'] = self.extract_shipping_address(combined_text)
        
        return extracted
    
    def extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number"""
        return self._extract_with_patterns(text, self.patterns['invoice_number'])
    
    def extract_invoice_date(self, text: str) -> Optional[str]:
        """Extract invoice date"""
        return self._extract_with_patterns(text, self.patterns['invoice_date'])
    
    def extract_gst_number(self, text: str, type_: str) -> Optional[str]:
        """Extract GST number"""
        return self._extract_with_patterns(text, self.patterns['gst_number'])
    
    def extract_po_number(self, text: str) -> Optional[str]:
        """Extract PO number"""
        return self._extract_with_patterns(text, self.patterns['po_number'])
    
    def extract_shipping_address(self, text: str) -> Optional[str]:
        """Extract shipping address (basic implementation)"""
        # Look for address patterns
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'ship' in line.lower() and 'to' in line.lower():
                # Try to get next few lines as address
                address_lines = lines[i+1:i+4]
                return '\n'.join(address_lines).strip()
        return None
    
    def _extract_with_patterns(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract field using regex patterns"""
        text_lower = text.lower()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None


# ==========================================
# Additional placeholder modules
# ==========================================

class TableParser:
    """Parses table data from invoices"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_line_items(self, images, ocr_results):
        """Extract line items from table"""
        # Placeholder implementation
        return []

class SealDetector:
    """Detects and extracts seals/signatures"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_and_extract_seals(self, images, pdf_path):
        """Detect seals and signatures"""
        # Placeholder implementation
        return {'detected': False, 'image_paths': []}

class Verifier:
    """Verifies and validates extracted data"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def verify_invoice_data(self, data):
        """Verify invoice data"""
        # Placeholder implementation
        return {'verified': True}

class OutputGenerator:
    """Generates output files"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_all_outputs(self, data, verification, pdf_path):
        """Generate all output files"""
        # Placeholder implementation
        return {'json': 'output/extracted_data.json'}