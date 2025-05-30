"""
Test module for Field Extractor functionality
Tests extraction of specific invoice fields using pattern matching and regex
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
import re
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules.field_extractor import FieldExtractor


class TestFieldExtractor(unittest.TestCase):
    """Test cases for Field Extractor module"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.field_extractor = FieldExtractor()
        
        # Mock OCR text data
        self.mock_ocr_text = """
        INVOICE
        Invoice Number: INV-2024-001
        Invoice Date: 15/05/2024
        
        Supplier Details:
        GST Number: 07AABCU9603R1ZV
        
        Bill To:
        Customer GST: 29GGGGG1314R9Z6
        
        PO Number: PO-2024-789
        
        Shipping Address:
        123 Main Street
        Bangalore, Karnataka
        PIN: 560001
        
        Item Details:
        S.No | Description | HSN/SAC | Qty | Unit Price | Total
        1 | Software License | 998311 | 2 | 5000.00 | 10000.00
        2 | Support Services | 998314 | 1 | 2500.00 | 2500.00
        
        Subtotal: 12500.00
        CGST (9%): 1125.00
        SGST (9%): 1125.00
        Total Amount: 14750.00
        """
        
        # Mock OCR coordinates data
        self.mock_coordinates = [
            {'text': 'INVOICE', 'bbox': (100, 50, 200, 80), 'confidence': 95},
            {'text': 'Invoice', 'bbox': (50, 100, 120, 130), 'confidence': 90},
            {'text': 'Number:', 'bbox': (130, 100, 200, 130), 'confidence': 88},
            {'text': 'INV-2024-001', 'bbox': (210, 100, 320, 130), 'confidence': 92},
            {'text': 'Invoice', 'bbox': (50, 150, 120, 180), 'confidence': 90},
            {'text': 'Date:', 'bbox': (130, 150, 180, 180), 'confidence': 89},
            {'text': '15/05/2024', 'bbox': (190, 150, 280, 180), 'confidence': 91},
            {'text': 'GST', 'bbox': (50, 250, 80, 280), 'confidence': 88},
            {'text': 'Number:', 'bbox': (90, 250, 150, 280), 'confidence': 85},
            {'text': '07AABCU9603R1ZV', 'bbox': (160, 250, 320, 280), 'confidence': 93}
        ]

    def test_extract_invoice_number_standard_format(self):
        """Test extraction of invoice number in standard formats"""
        test_cases = [
            ("Invoice Number: INV-2024-001", "INV-2024-001"),
            ("INVOICE NO: 12345", "12345"),
            ("Inv No INV/2024/0001", "INV/2024/0001"),
            ("Bill No. B-2024-789", "B-2024-789")
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.field_extractor.extract_invoice_number(text, [])
                self.assertEqual(result['value'], expected)
                self.assertGreater(result['confidence'], 0.8)

    def test_extract_invoice_number_with_coordinates(self):
        """Test invoice number extraction using coordinate information"""
        result = self.field_extractor.extract_invoice_number(
            self.mock_ocr_text, self.mock_coordinates
        )
        
        self.assertEqual(result['value'], 'INV-2024-001')
        self.assertGreater(result['confidence'], 0.8)
        self.assertIn('bbox', result)

    def test_extract_dates_multiple_formats(self):
        """Test date extraction in various formats"""
        test_cases = [
            ("Invoice Date: 15/05/2024", "15/05/2024"),
            ("Date: 15-May-2024", "15-May-2024"),
            ("Invoice Date 2024-05-15", "2024-05-15"),
            ("Date: May 15, 2024", "May 15, 2024"),
            ("15.05.2024", "15.05.2024")
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.field_extractor.extract_dates(text, [])
                self.assertEqual(result['value'], expected)
                self.assertGreater(result['confidence'], 0.7)

    def test_extract_gst_numbers_valid_formats(self):
        """Test GST number extraction with validation"""
        test_cases = [
            ("GST: 07AABCU9603R1ZV", "07AABCU9603R1ZV"),
            ("GSTIN 29GGGGG1314R9Z6", "29GGGGG1314R9Z6"),
            ("GST Number: 27AAPFU0939F1ZV", "27AAPFU0939F1ZV")
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.field_extractor.extract_gst_numbers(text, [])
                self.assertIn(expected, result['supplier_gst'] or result['bill_to_gst'])

    def test_extract_gst_numbers_invalid_format(self):
        """Test GST number extraction with invalid formats"""
        invalid_gst_text = "GST: INVALID123"
        result = self.field_extractor.extract_gst_numbers(invalid_gst_text, [])
        
        # Should not extract invalid GST numbers
        self.assertIsNone(result['supplier_gst'])
        self.assertIsNone(result['bill_to_gst'])

    def test_extract_po_number(self):
        """Test PO number extraction"""
        test_cases = [
            ("PO Number: PO-2024-789", "PO-2024-789"),
            ("Purchase Order: 12345", "12345"),
            ("PO No PO/2024/001", "PO/2024/001"),
            ("P.O. Number: PO_2024_456", "PO_2024_456")
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.field_extractor.extract_po_number(text, [])
                self.assertEqual(result['value'], expected)
                self.assertGreater(result['confidence'], 0.8)

    def test_extract_shipping_address_multiline(self):
        """Test shipping address extraction from multiline text"""
        address_text = """
        Shipping Address:
        ABC Company Pvt Ltd
        123 Main Street, Block A
        Bangalore, Karnataka
        PIN: 560001
        India
        """
        
        result = self.field_extractor.extract_shipping_address(address_text, [])
        
        self.assertIn('123 Main Street', result['value'])
        self.assertIn('Bangalore', result['value'])
        self.assertIn('560001', result['value'])
        self.assertGreater(result['confidence'], 0.7)

    def test_extract_shipping_address_single_line(self):
        """Test shipping address extraction from single line"""
        address_text = "Ship To: 456 Park Avenue, Mumbai, Maharashtra - 400001"
        
        result = self.field_extractor.extract_shipping_address(address_text, [])
        
        self.assertIn('456 Park Avenue', result['value'])
        self.assertIn('Mumbai', result['value'])
        self.assertIn('400001', result['value'])

    def test_calculate_field_confidence_high_ocr_confidence(self):
        """Test field confidence calculation with high OCR confidence"""
        field_text = "INV-2024-001"
        ocr_confidence = 0.95
        pattern_match_strength = 1.0  # Perfect pattern match
        
        confidence = self.field_extractor.calculate_field_confidence(
            field_text, ocr_confidence, pattern_match_strength
        )
        
        self.assertGreater(confidence, 0.9)
        self.assertLessEqual(confidence, 1.0)

    def test_calculate_field_confidence_low_ocr_confidence(self):
        """Test field confidence calculation with low OCR confidence"""
        field_text = "unclear_text"
        ocr_confidence = 0.3
        pattern_match_strength = 0.5
        
        confidence = self.field_extractor.calculate_field_confidence(
            field_text, ocr_confidence, pattern_match_strength
        )
        
        self.assertLess(confidence, 0.5)

    def test_validate_extracted_field_valid_invoice_number(self):
        """Test validation of extracted invoice number"""
        valid_invoice_numbers = [
            "INV-2024-001",
            "12345",
            "INV/2024/0001",
            "B-2024-789"
        ]
        
        for inv_num in valid_invoice_numbers:
            with self.subTest(invoice_number=inv_num):
                is_valid = self.field_extractor.validate_extracted_field(
                    inv_num, 'invoice_number'
                )
                self.assertTrue(is_valid)

    def test_validate_extracted_field_valid_gst_number(self):
        """Test validation of extracted GST number"""
        valid_gst_numbers = [
            "07AABCU9603R1ZV",
            "29GGGGG1314R9Z6",
            "27AAPFU0939F1ZV"
        ]
        
        for gst_num in valid_gst_numbers:
            with self.subTest(gst_number=gst_num):
                is_valid = self.field_extractor.validate_extracted_field(
                    gst_num, 'gst_number'
                )
                self.assertTrue(is_valid)

    def test_validate_extracted_field_invalid_gst_number(self):
        """Test validation of invalid GST number"""
        invalid_gst_numbers = [
            "INVALID123",
            "07AABCU9603R1Z",  # Too short
            "07AABCU9603R1ZVX",  # Too long
            "ABCDEFGHIJKLMNO"  # Wrong format
        ]
        
        for gst_num in invalid_gst_numbers:
            with self.subTest(gst_number=gst_num):
                is_valid = self.field_extractor.validate_extracted_field(
                    gst_num, 'gst_number'
                )
                self.assertFalse(is_valid)

    def test_extract_all_fields_comprehensive(self):
        """Test extraction of all fields from comprehensive text"""
        result = self.field_extractor.extract_all_fields(
            self.mock_ocr_text, self.mock_coordinates
        )
        
        # Check that all expected fields are present
        expected_fields = [
            'invoice_number', 'invoice_date', 'supplier_gst_number',
            'bill_to_gst_number', 'po_number', 'shipping_address'
        ]
        
        for field in expected_fields:
            self.assertIn(field, result)
            self.assertIsNotNone(result[field]['value'])
            self.assertGreater(result[field]['confidence'], 0.5)

    def test_extract_field_with_context_window(self):
        """Test field extraction using context window around keywords"""
        text = "Additional charges: Invoice Number INV-2024-999 for services"
        
        result = self.field_extractor.extract_field_with_context(
            text, 'invoice_number', context_window=50
        )
        
        self.assertEqual(result['value'], 'INV-2024-999')
        self.assertGreater(result['confidence'], 0.8)

    def test_extract_field_nearest_to_keyword(self):
        """Test extraction of field value nearest to keyword"""
        coordinates = [
            {'text': 'Invoice', 'bbox': (50, 100, 120, 130), 'confidence': 90},
            {'text': 'Number:', 'bbox': (130, 100, 200, 130), 'confidence': 88},
            {'text': 'INV-2024-001', 'bbox': (210, 100, 320, 130), 'confidence': 92},
            {'text': 'WRONG-NUMBER', 'bbox': (400, 100, 520, 130), 'confidence': 85}
        ]
        
        result = self.field_extractor.extract_field_nearest_to_keyword(
            coordinates, ['Invoice', 'Number'], field_type='invoice_number'
        )
        
        self.assertEqual(result['value'], 'INV-2024-001')
        self.assertNotEqual(result['value'], 'WRONG-NUMBER')

    def test_preprocess_text_for_extraction(self):
        """Test text preprocessing for better extraction"""
        messy_text = "  Invoice    Number:   INV-2024-001  \n\n  "
        
        cleaned_text = self.field_extractor.preprocess_text_for_extraction(messy_text)
        
        self.assertEqual(cleaned_text, "Invoice Number: INV-2024-001")

    def test_extract_field_from_table_structure(self):
        """Test field extraction from table-like structure"""
        table_text = """
        Field               Value
        Invoice Number      INV-2024-001
        Invoice Date        15/05/2024
        GST Number          07AABCU9603R1ZV
        """
        
        result = self.field_extractor.extract_field_from_table_structure(
            table_text, 'Invoice Number'
        )
        
        self.assertEqual(result['value'], 'INV-2024-001')
        self.assertGreater(result['confidence'], 0.8)

    def test_error_handling_empty_text(self):
        """Test error handling for empty text input"""
        result = self.field_extractor.extract_invoice_number("", [])
        
        self.assertIsNone(result['value'])
        self.assertEqual(result['confidence'], 0.0)

    def test_error_handling_none_input(self):
        """Test error handling for None input"""
        result = self.field_extractor.extract_invoice_number(None, [])
        
        self.assertIsNone(result['value'])
        self.assertEqual(result['confidence'], 0.0)

    def test_pattern_matching_strength_calculation(self):
        """Test pattern matching strength calculation"""
        # Perfect match
        perfect_match = self.field_extractor.calculate_pattern_match_strength(
            "INV-2024-001", r'INV-\d{4}-\d{3}'
        )
        self.assertEqual(perfect_match, 1.0)
        
        # Partial match
        partial_match = self.field_extractor.calculate_pattern_match_strength(
            "INV-2024", r'INV-\d{4}-\d{3}'
        )
        self.assertLess(partial_match, 1.0)
        self.assertGreater(partial_match, 0.0)
        
        # No match
        no_match = self.field_extractor.calculate_pattern_match_strength(
            "RANDOM-TEXT", r'INV-\d{4}-\d{3}'
        )
        self.assertEqual(no_match, 0.0)

    def test_date_format_standardization(self):
        """Test standardization of different date formats"""
        date_formats = [
            ("15/05/2024", "15/05/2024"),
            ("15-May-2024", "15/05/2024"),
            ("2024-05-15", "15/05/2024"),
            ("May 15, 2024", "15/05/2024")
        ]
        
        for input_date, expected_output in date_formats:
            with self.subTest(input_date=input_date):
                standardized = self.field_extractor.standardize_date_format(input_date)
                self.assertEqual(standardized, expected_output)

    def test_confidence_threshold_filtering(self):
        """Test filtering of fields based on confidence threshold"""
        low_confidence_result = {
            'invoice_number': {'value': 'INV-001', 'confidence': 0.3},
            'invoice_date': {'value': '15/05/2024', 'confidence': 0.9},
            'gst_number': {'value': '07AABCU9603R1ZV', 'confidence': 0.4}
        }
        
        filtered_result = self.field_extractor.filter_by_confidence_threshold(
            low_confidence_result, threshold=0.5
        )
        
        # Only high confidence fields should remain
        self.assertNotIn('invoice_number', filtered_result)
        self.assertIn('invoice_date', filtered_result)
        self.assertNotIn('gst_number', filtered_result)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFieldExtractor)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")