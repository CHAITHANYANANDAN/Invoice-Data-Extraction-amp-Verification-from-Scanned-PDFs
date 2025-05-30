"""
Module 4: Field Extractor (field_extractor.py)
Purpose: Extract specific invoice fields from OCR text using pattern matching
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ExtractedField:
    """Data class to hold extracted field information"""
    value: str
    confidence: float
    coordinates: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    source_text: str  # Original text from which field was extracted

class FieldExtractor:
    """
    Extracts specific invoice fields from OCR text using pattern matching
    """
    
    def __init__(self):
        """Initialize field extractor with patterns and configurations"""
        self.invoice_patterns = [
            r'(?:invoice\s*(?:no|number|#)[\s:]*)([\w\-/]+)',
            r'(?:inv\s*(?:no|#)[\s:]*)([\w\-/]+)',
            r'(?:bill\s*(?:no|number|#)[\s:]*)([\w\-/]+)',
            r'(?:receipt\s*(?:no|number|#)[\s:]*)([\w\-/]+)'
        ]
        
        self.date_patterns = [
            r'(?:date[\s:]*)((?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}))',
            r'(?:invoice\s*date[\s:]*)((?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}))',
            r'(?:bill\s*date[\s:]*)((?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}))',
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})'
        ]
        
        self.gst_patterns = [
            r'(?:gstin[\s:]*)((?:\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})|(?:[A-Z0-9]{15}))',
            r'(?:gst\s*(?:no|number)[\s:]*)((?:\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})|(?:[A-Z0-9]{15}))',
            r'(?:tax\s*(?:id|number)[\s:]*)((?:\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})|(?:[A-Z0-9]{15}))',
            r'(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})'
        ]
        
        self.po_patterns = [
            r'(?:po\s*(?:no|number)[\s:]*)([\w\-/]+)',
            r'(?:purchase\s*order[\s:]*)([\w\-/]+)',
            r'(?:order\s*(?:no|number)[\s:]*)([\w\-/]+)',
            r'(?:ref\s*(?:no|number)[\s:]*)([\w\-/]+)'
        ]
        
        self.amount_patterns = [
            r'(?:total[\s:]*(?:amount)?[\s:]*(?:rs\.?|₹)?[\s]*)([\d,]+\.?\d*)',
            r'(?:grand\s*total[\s:]*(?:rs\.?|₹)?[\s]*)([\d,]+\.?\d*)',
            r'(?:final\s*amount[\s:]*(?:rs\.?|₹)?[\s]*)([\d,]+\.?\d*)',
            r'(?:net\s*amount[\s:]*(?:rs\.?|₹)?[\s]*)([\d,]+\.?\d*)'
        ]
        
        # Address indicators
        self.address_indicators = [
            'ship to', 'shipping address', 'deliver to', 'delivery address',
            'bill to', 'billing address', 'address', 'consignee'
        ]
        
    def extract_invoice_number(self, text: str, coordinates: List[Dict]) -> Optional[ExtractedField]:
        """
        Extract invoice number from OCR text
        
        Args:
            text (str): OCR extracted text
            coordinates (List[Dict]): OCR results with coordinates and confidence
            
        Returns:
            Optional[ExtractedField]: Extracted invoice number with metadata
        """
        try:
            text_lower = text.lower()
            
            for pattern in self.invoice_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    invoice_num = match.group(1).strip()
                    if len(invoice_num) > 2:  # Minimum length check
                        # Find coordinates of the matched text
                        coords, conf, source = self._find_text_coordinates(
                            match.group(0), coordinates
                        )
                        
                        return ExtractedField(
                            value=invoice_num.upper(),
                            confidence=conf,
                            coordinates=coords,
                            source_text=source
                        )
                        
            logger.warning("No invoice number found")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting invoice number: {str(e)}")
            return None
    
    def extract_dates(self, text: str, coordinates: List[Dict]) -> Optional[ExtractedField]:
        """
        Extract invoice date from OCR text
        
        Args:
            text (str): OCR extracted text
            coordinates (List[Dict]): OCR results with coordinates and confidence
            
        Returns:
            Optional[ExtractedField]: Extracted date with metadata
        """
        try:
            text_lower = text.lower()
            
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    date_str = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                    
                    # Validate and normalize date
                    normalized_date = self._normalize_date(date_str)
                    if normalized_date:
                        coords, conf, source = self._find_text_coordinates(
                            match.group(0), coordinates
                        )
                        
                        return ExtractedField(
                            value=normalized_date,
                            confidence=conf,
                            coordinates=coords,
                            source_text=source
                        )
                        
            logger.warning("No valid date found")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting date: {str(e)}")
            return None
    
    def extract_gst_numbers(self, text: str, coordinates: List[Dict]) -> List[ExtractedField]:
        """
        Extract GST numbers from OCR text
        
        Args:
            text (str): OCR extracted text
            coordinates (List[Dict]): OCR results with coordinates and confidence
            
        Returns:
            List[ExtractedField]: List of extracted GST numbers with metadata
        """
        try:
            gst_numbers = []
            text_upper = text.upper()
            
            for pattern in self.gst_patterns:
                matches = re.finditer(pattern, text_upper, re.IGNORECASE)
                for match in matches:
                    gst_num = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                    
                    # Validate GST format
                    if self._validate_gst_format(gst_num):
                        coords, conf, source = self._find_text_coordinates(
                            match.group(0), coordinates
                        )
                        
                        gst_numbers.append(ExtractedField(
                            value=gst_num.upper(),
                            confidence=conf,
                            coordinates=coords,
                            source_text=source
                        ))
                        
            return gst_numbers
            
        except Exception as e:
            logger.error(f"Error extracting GST numbers: {str(e)}")
            return []
    
    def extract_po_number(self, text: str, coordinates: List[Dict]) -> Optional[ExtractedField]:
        """
        Extract Purchase Order number from OCR text
        
        Args:
            text (str): OCR extracted text
            coordinates (List[Dict]): OCR results with coordinates and confidence
            
        Returns:
            Optional[ExtractedField]: Extracted PO number with metadata
        """
        try:
            text_lower = text.lower()
            
            for pattern in self.po_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    po_num = match.group(1).strip()
                    if len(po_num) > 2:  # Minimum length check
                        coords, conf, source = self._find_text_coordinates(
                            match.group(0), coordinates
                        )
                        
                        return ExtractedField(
                            value=po_num.upper(),
                            confidence=conf,
                            coordinates=coords,
                            source_text=source
                        )
                        
            logger.info("No PO number found")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting PO number: {str(e)}")
            return None
    
    def extract_shipping_address(self, text: str, coordinates: List[Dict]) -> Optional[ExtractedField]:
        """
        Extract shipping address from OCR text
        
        Args:
            text (str): OCR extracted text
            coordinates (List[Dict]): OCR results with coordinates and confidence
            
        Returns:
            Optional[ExtractedField]: Extracted shipping address with metadata
        """
        try:
            text_lower = text.lower()
            lines = text.split('\n')
            
            # Find address section
            address_start_idx = -1
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                for indicator in self.address_indicators:
                    if indicator in line_lower:
                        address_start_idx = i
                        break
                if address_start_idx != -1:
                    break
            
            if address_start_idx == -1:
                logger.warning("No shipping address section found")
                return None
            
            # Extract address lines (typically next 3-5 lines after indicator)
            address_lines = []
            for i in range(address_start_idx + 1, min(address_start_idx + 6, len(lines))):
                if i < len(lines):
                    line = lines[i].strip()
                    if line and not self._is_likely_end_of_address(line):
                        address_lines.append(line)
                    else:
                        break
            
            if address_lines:
                full_address = '\n'.join(address_lines)
                
                # Find approximate coordinates
                coords, conf, source = self._find_text_coordinates(
                    ' '.join(address_lines[:2]), coordinates  # Use first two lines for coordinate search
                )
                
                return ExtractedField(
                    value=full_address,
                    confidence=conf,
                    coordinates=coords,
                    source_text=source
                )
                
            logger.warning("No valid shipping address found")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting shipping address: {str(e)}")
            return None
    
    def extract_total_amount(self, text: str, coordinates: List[Dict]) -> Optional[ExtractedField]:
        """
        Extract total amount from OCR text
        
        Args:
            text (str): OCR extracted text
            coordinates (List[Dict]): OCR results with coordinates and confidence
            
        Returns:
            Optional[ExtractedField]: Extracted total amount with metadata
        """
        try:
            text_lower = text.lower()
            
            for pattern in self.amount_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    amount_str = match.group(1).strip()
                    
                    # Clean and validate amount
                    cleaned_amount = self._clean_amount(amount_str)
                    if cleaned_amount:
                        coords, conf, source = self._find_text_coordinates(
                            match.group(0), coordinates
                        )
                        
                        return ExtractedField(
                            value=cleaned_amount,
                            confidence=conf,
                            coordinates=coords,
                            source_text=source
                        )
                        
            logger.warning("No total amount found")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting total amount: {str(e)}")
            return None
    
    def calculate_field_confidence(self, field_text: str, ocr_confidence: float) -> float:
        """
        Calculate confidence score for extracted field
        
        Args:
            field_text (str): Extracted field text
            ocr_confidence (float): OCR confidence score
            
        Returns:
            float: Calculated confidence score (0.0 to 1.0)
        """
        try:
            # Base confidence from OCR
            base_confidence = ocr_confidence
            
            # Adjust based on field characteristics
            confidence_adjustments = 0.0
            
            # Length-based adjustment
            if len(field_text) >= 5:
                confidence_adjustments += 0.1
            elif len(field_text) <= 2:
                confidence_adjustments -= 0.2
            
            # Pattern-based adjustment (alphanumeric is generally more reliable)
            if re.match(r'^[A-Za-z0-9\-/]+$', field_text):
                confidence_adjustments += 0.1
            
            # Special character penalty
            if re.search(r'[^\w\s\-/.,]', field_text):
                confidence_adjustments -= 0.1
            
            final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustments))
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating field confidence: {str(e)}")
            return 0.5  # Default confidence
    
    def _find_text_coordinates(self, search_text: str, coordinates: List[Dict]) -> Tuple[Tuple[int, int, int, int], float, str]:
        """
        Find coordinates of text in OCR results
        
        Args:
            search_text (str): Text to search for
            coordinates (List[Dict]): OCR results with coordinates
            
        Returns:
            Tuple: (coordinates, confidence, source_text)
        """
        try:
            search_lower = search_text.lower().strip()
            best_match = None
            best_confidence = 0.0
            
            for result in coordinates:
                text_lower = result.get('text', '').lower().strip()
                
                if search_lower in text_lower or text_lower in search_lower:
                    confidence = result.get('confidence', 0.0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = result
            
            if best_match:
                bbox = best_match.get('bbox', (0, 0, 0, 0))
                return bbox, best_confidence, best_match.get('text', '')
            else:
                return (0, 0, 0, 0), 0.5, search_text
                
        except Exception as e:
            logger.error(f"Error finding text coordinates: {str(e)}")
            return (0, 0, 0, 0), 0.5, search_text
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normalize date string to standard format
        
        Args:
            date_str (str): Raw date string
            
        Returns:
            Optional[str]: Normalized date in YYYY-MM-DD format
        """
        try:
            # Common date formats to try
            date_formats = [
                '%d/%m/%Y', '%d-%m-%Y', '%d/%m/%y', '%d-%m-%y',
                '%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y',
                '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y'
            ]
            
            date_str = date_str.strip()
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_str}")
            return None
            
        except Exception as e:
            logger.error(f"Error normalizing date: {str(e)}")
            return None
    
    def _validate_gst_format(self, gst_num: str) -> bool:
        """
        Validate GST number format
        
        Args:
            gst_num (str): GST number to validate
            
        Returns:
            bool: True if valid format
        """
        try:
            # Standard GST format: 15 characters
            if len(gst_num) != 15:
                return False
            
            # Pattern: 2 digits + 5 letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric
            pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z0-9]{1}[Z]{1}[A-Z0-9]{1}$'
            return bool(re.match(pattern, gst_num))
            
        except Exception as e:
            logger.error(f"Error validating GST format: {str(e)}")
            return False
    
    def _clean_amount(self, amount_str: str) -> Optional[str]:
        """
        Clean and validate amount string
        
        Args:
            amount_str (str): Raw amount string
            
        Returns:
            Optional[str]: Cleaned amount string
        """
        try:
            # Remove currency symbols and extra spaces
            cleaned = re.sub(r'[₹$€£,\s]', '', amount_str)
            
            # Validate numeric format
            if re.match(r'^\d+\.?\d*$', cleaned):
                return cleaned
            
            return None
            
        except Exception as e:
            logger.error(f"Error cleaning amount: {str(e)}")
            return None
    
    def _is_likely_end_of_address(self, line: str) -> bool:
        """
        Check if line is likely the end of an address section
        
        Args:
            line (str): Text line to check
            
        Returns:
            bool: True if likely end of address
        """
        try:
            line_lower = line.lower().strip()
            
            # Keywords that indicate end of address
            end_indicators = [
                'phone', 'tel', 'mobile', 'email', 'website', 'gstin', 'gst',
                'invoice', 'bill', 'date', 'po number', 'order'
            ]
            
            for indicator in end_indicators:
                if indicator in line_lower:
                    return True
            
            # Very short lines might be end of address
            if len(line.strip()) < 3:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking end of address: {str(e)}")
            return False


# Example usage and testing functions
def test_field_extractor():
    """Test function for field extractor"""
    
    # Sample OCR data
    sample_text = """
    INVOICE
    Invoice No: INV-2024-001
    Date: 15/03/2024
    
    Bill To:
    ABC Company Ltd
    123 Business Street
    Mumbai, Maharashtra 400001
    
    Ship To:
    XYZ Corp
    456 Industrial Avenue
    Delhi, Delhi 110001
    
    GSTIN: 27AABCU9603R1ZM
    PO Number: PO-12345
    
    Total Amount: Rs. 15,750.00
    """
    
    # Mock coordinates data
    mock_coordinates = [
        {'text': 'Invoice No: INV-2024-001', 'confidence': 0.95, 'bbox': (100, 50, 300, 70)},
        {'text': 'Date: 15/03/2024', 'confidence': 0.92, 'bbox': (100, 80, 250, 100)},
        {'text': 'GSTIN: 27AABCU9603R1ZM', 'confidence': 0.88, 'bbox': (100, 200, 350, 220)},
        {'text': 'PO Number: PO-12345', 'confidence': 0.90, 'bbox': (100, 230, 280, 250)},
        {'text': 'Total Amount: Rs. 15,750.00', 'confidence': 0.93, 'bbox': (100, 400, 320, 420)}
    ]
    
    extractor = FieldExtractor()
    
    # Test extractions
    print("=== Field Extraction Test ===")
    
    invoice_num = extractor.extract_invoice_number(sample_text, mock_coordinates)
    if invoice_num:
        print(f"Invoice Number: {invoice_num.value} (Confidence: {invoice_num.confidence:.2f})")
    
    date = extractor.extract_dates(sample_text, mock_coordinates)
    if date:
        print(f"Date: {date.value} (Confidence: {date.confidence:.2f})")
    
    gst_numbers = extractor.extract_gst_numbers(sample_text, mock_coordinates)
    for gst in gst_numbers:
        print(f"GST Number: {gst.value} (Confidence: {gst.confidence:.2f})")
    
    po_number = extractor.extract_po_number(sample_text, mock_coordinates)
    if po_number:
        print(f"PO Number: {po_number.value} (Confidence: {po_number.confidence:.2f})")
    
    total_amount = extractor.extract_total_amount(sample_text, mock_coordinates)
    if total_amount:
        print(f"Total Amount: {total_amount.value} (Confidence: {total_amount.confidence:.2f})")


if __name__ == "__main__":
    test_field_extractor()