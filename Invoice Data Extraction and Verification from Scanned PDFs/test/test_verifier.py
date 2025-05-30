"""
Test module for Verifier functionality
Tests data verification, validation, mathematical checks, and confidence reporting
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
from decimal import Decimal, ROUND_HALF_UP

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules.verifier import Verifier


class TestVerifier(unittest.TestCase):
    """Test cases for Verifier module"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.verifier = Verifier()
        
        # Mock line items data
        self.mock_line_items = [
            {
                'serial_number': 1,
                'description': 'Software License',
                'hsn_sac': '998311',
                'quantity': 2,
                'unit_price': 5000.00,
                'total_amount': 10000.00,
                'confidence_scores': {
                    'description': 0.95,
                    'hsn_sac': 0.88,
                    'quantity': 0.92,
                    'unit_price': 0.94,
                    'total_amount': 0.90
                }
            },
            {
                'serial_number': 2,
                'description': 'Support Services',
                'hsn_sac': '998314',
                'quantity': 1,
                'unit_price': 2500.00,
                'total_amount': 2500.00,
                'confidence_scores': {
                    'description': 0.93,
                    'hsn_sac': 0.85,
                    'quantity': 0.96,
                    'unit_price': 0.91,
                    'total_amount': 0.89
                }
            }
        ]
        
        # Mock invoice totals
        self.mock_totals = {
            'subtotal': 12500.00,
            'discount': 0.00,
            'cgst': 1125.00,
            'sgst': 1125.00,
            'igst': 0.00,
            'total_gst': 2250.00,
            'final_total': 14750.00
        }
        
        # Mock extracted fields
        self.mock_extracted_fields = {
            'invoice_number': {'value': 'INV-2024-001', 'confidence': 0.94},
            'invoice_date': {'value': '15/05/2024', 'confidence': 0.91},
            'supplier_gst_number': {'value': '07AABCU9603R1ZV', 'confidence': 0.89},
            'bill_to_gst_number': {'value': '29GGGGG1314R9Z6', 'confidence': 0.88},
            'po_number': {'value': 'PO-2024-789', 'confidence': 0.86},
            'shipping_address': {'value': '123 Main St, Bangalore', 'confidence': 0.90},
            'seal_and_sign_present': {'value': True, 'confidence': 0.80}
        }

    def test_verify_line_item_calculations_correct(self):
        """Test verification of correct line item calculations"""
        verification_results = self.verifier.verify_line_item_calculations(self.mock_line_items)
        
        # All calculations should pass
        for i, result in enumerate(verification_results):
            with self.subTest(line_item=i+1):
                self.assertTrue(result['line_total_check']['check_passed'])
                self.assertEqual(result['calculated_value'], result['extracted_value'])

    def test_verify_line_item_calculations_incorrect(self):
        """Test verification of incorrect line item calculations"""
        incorrect_line_items = [
            {
                'serial_number': 1,
                'description': 'Test Item',
                'quantity': 2,
                'unit_price': 100.00,
                'total_amount': 150.00,  # Incorrect: should be 200.00
                'confidence_scores': {
                    'quantity': 0.95,
                    'unit_price': 0.93,
                    'total_amount': 0.88
                }
            }
        ]
        
        verification_results = self.verifier.verify_line_item_calculations(incorrect_line_items)
        
        self.assertFalse(verification_results[0]['line_total_check']['check_passed'])
        self.assertEqual(verification_results[0]['calculated_value'], 200.00)
        self.assertEqual(verification_results[0]['extracted_value'], 150.00)

    def test_verify_subtotal_calculation_correct(self):
        """Test verification of correct subtotal calculation"""
        result = self.verifier.verify_subtotal_calculation(
            self.mock_line_items, self.mock_totals['subtotal']
        )
        
        self.assertTrue(result['check_passed'])
        self.assertEqual(result['calculated_value'], 12500.00)
        self.assertEqual(result['extracted_value'], 12500.00)

    def test_verify_subtotal_calculation_incorrect(self):
        """Test verification of incorrect subtotal calculation"""
        incorrect_subtotal = 10000.00  # Should be 12500.00
        
        result = self.verifier.verify_subtotal_calculation(
            self.mock_line_items, incorrect_subtotal
        )
        
        self.assertFalse(result['check_passed'])
        self.assertEqual(result['calculated_value'], 12500.00)
        self.assertEqual(result['extracted_value'], 10000.00)

    def test_verify_final_total_calculation_correct(self):
        """Test verification of correct final total calculation"""
        result = self.verifier.verify_final_total(
            subtotal=self.mock_totals['subtotal'],
            discount=self.mock_totals['discount'],
            gst=self.mock_totals['total_gst'],
            final_total=self.mock_totals['final_total']
        )
        
        self.assertTrue(result['check_passed'])
        self.assertEqual(result['calculated_value'], 14750.00)
        self.assertEqual(result['extracted_value'], 14750.00)

    def test_verify_final_total_calculation_incorrect(self):
        """Test verification of incorrect final total calculation"""
        incorrect_final_total = 15000.00  # Should be 14750.00
        
        result = self.verifier.verify_final_total(
            subtotal=self.mock_totals['subtotal'],
            discount=self.mock_totals['discount'],
            gst=self.mock_totals['total_gst'],
            final_total=incorrect_final_total
        )
        
        self.assertFalse(result['check_passed'])
        self.assertEqual(result['calculated_value'], 14750.00)
        self.assertEqual(result['extracted_value'], 15000.00)

    def test_verify_gst_calculations_cgst_sgst(self):
        """Test verification of CGST and SGST calculations"""
        gst_details = {
            'cgst_rate': 9.0,
            'cgst_amount': 1125.00,
            'sgst_rate': 9.0,
            'sgst_amount': 1125.00,
            'igst_rate': 0.0,
            'igst_amount': 0.00
        }
        
        result = self.verifier.verify_gst_calculations(
            subtotal=self.mock_totals['subtotal'],
            gst_details=gst_details
        )
        
        self.assertTrue(result['cgst_check']['check_passed'])
        self.assertTrue(result['sgst_check']['check_passed'])
        self.assertEqual(result['total_gst_calculated'], 2250.00)

    def test_verify_gst_calculations_igst(self):
        """Test verification of IGST calculations"""
        gst_details = {
            'cgst_rate': 0.0,
            'cgst_amount': 0.00,
            'sgst_rate': 0.0,
            'sgst_amount': 0.00,
            'igst_rate': 18.0,
            'igst_amount': 2250.00
        }
        
        result = self.verifier.verify_gst_calculations(
            subtotal=self.mock_totals['subtotal'],
            gst_details=gst_details
        )
        
        self.assertTrue(result['igst_check']['check_passed'])
        self.assertEqual(result['total_gst_calculated'], 2250.00)

    def test_generate_verification_flags_all_passed(self):
        """Test generation of verification flags when all checks pass"""
        line_item_results = [
            {'line_total_check': {'check_passed': True}},
            {'line_total_check': {'check_passed': True}}
        ]
        
        calculation_results = {
            'subtotal_check': {'check_passed': True},
            'final_total_check': {'check_passed': True},
            'gst_check': {'check_passed': True}
        }
        
        flags = self.verifier.generate_verification_flags(
            line_item_results, calculation_results
        )
        
        self.assertTrue(flags['all_line_items_verified'])
        self.assertTrue(flags['totals_verified'])
        self.assertTrue(flags['overall_verification_passed'])
        self.assertEqual(len(flags['failed_checks']), 0)

    def test_generate_verification_flags_some_failed(self):
        """Test generation of verification flags when some checks fail"""
        line_item_results = [
            {'line_total_check': {'check_passed': True}},
            {'line_total_check': {'check_passed': False}}
        ]
        
        calculation_results = {
            'subtotal_check': {'check_passed': False},
            'final_total_check': {'check_passed': True},
            'gst_check': {'check_passed': True}
        }
        
        flags = self.verifier.generate_verification_flags(
            line_item_results, calculation_results
        )
        
        self.assertFalse(flags['all_line_items_verified'])
        self.assertFalse(flags['totals_verified'])
        self.assertFalse(flags['overall_verification_passed'])
        self.assertGreater(len(flags['failed_checks']), 0)

    def test_create_confidence_report(self):
        """Test creation of comprehensive confidence report"""
        field_confidences = self.mock_extracted_fields
        line_item_confidences = self.mock_line_items
        
        report = self.verifier.create_confidence_report(
            field_confidences, line_item_confidences
        )
        
        # Check structure
        self.assertIn('field_verification', report)
        self.assertIn('line_items_verification', report)
        self.assertIn('summary', report)
        
        # Check field verification
        for field_name in self.mock_extracted_fields.keys():
            self.assertIn(field_name, report['field_verification'])
            self.assertIn('confidence', report['field_verification'][field_name])
            self.assertIn('present', report['field_verification'][field_name])
        
        # Check summary
        self.assertIn('all_fields_confident', report['summary'])
        self.assertIn('average_field_confidence', report['summary'])
        self.assertIn('low_confidence_fields', report['summary'])

    def test_validate_field_presence_all_present(self):
        """Test validation when all required fields are present"""
        validation_result = self.verifier.validate_field_presence(self.mock_extracted_fields)
        
        self.assertTrue(validation_result['all_required_fields_present'])
        self.assertEqual(len(validation_result['missing_fields']), 0)

    def test_validate_field_presence_some_missing(self):
        """Test validation when some required fields are missing"""
        incomplete_fields = {
            'invoice_number': {'value': 'INV-001', 'confidence': 0.9},
            'invoice_date': {'value': None, 'confidence': 0.0},
            # Missing other required fields
        }
        
        validation_result = self.verifier.validate_field_presence(incomplete_fields)
        
        self.assertFalse(validation_result['all_required_fields_present'])
        self.assertGreater(len(validation_result['missing_fields']), 0)
        self.assertIn('invoice_date', validation_result['missing_fields'])

    def test_calculate_verification_score(self):
        """Test calculation of overall verification score"""
        verification_data = {
            'field_verification_score': 0.90,
            'line_item_verification_score': 0.85,
            'calculation_verification_score': 0.95
        }
        
        overall_score = self.verifier.calculate_verification_score(verification_data)
        
        self.assertGreater(overall_score, 0.8)
        self.assertLessEqual(overall_score, 1.0)

    def test_identify_potential_ocr_errors(self):
        """Test identification of potential OCR errors"""
        low_confidence_line_items = [
            {
                'serial_number': 1,
                'description': 'unclear_text',
                'quantity': 2,
                'unit_price': 100.00,
                'total_amount': 200.00,
                'confidence_scores': {
                    'description': 0.3,  # Very low confidence
                    'quantity': 0.95,
                    'unit_price': 0.90,
                    'total_amount': 0.85
                }
            }
        ]
        
        potential_errors = self.verifier.identify_potential_ocr_errors(
            low_confidence_line_items, threshold=0.5
        )
        
        self.assertGreater(len(potential_errors), 0)
        self.assertIn('description', potential_errors[0]['low_confidence_fields'])

    def test_verify_invoice_number_format(self):
        """Test verification of invoice number format"""
        valid_formats = ['INV-2024-001', 'BILL/2024/001', '12345']
        invalid_formats = ['', 'INV', '???', 'INVALID@#$']
        
        for inv_num in valid_formats:
            with self.subTest(invoice_number=inv_num):
                result = self.verifier.verify_invoice_number_format(inv_num)
                self.assertTrue(result['format_valid'])
        
        for inv_num in invalid_formats:
            with self.subTest(invoice_number=inv_num):
                result = self.verifier.verify_invoice_number_format(inv_num)
                self.assertFalse(result['format_valid'])

    def test_verify_date_format(self):
        """Test verification of date format and validity"""
        valid_dates = ['15/05/2024', '2024-05-15', '15-May-2024']
        invalid_dates = ['32/13/2024', '2024-15-05', 'invalid_date', '']
        
        for date_str in valid_dates:
            with self.subTest(date=date_str):
                result = self.verifier.verify_date_format(date_str)
                self.assertTrue(result['format_valid'])
        
        for date_str in invalid_dates:
            with self.subTest(date=date_str):
                result = self.verifier.verify_date_format(date_str)
                self.assertFalse(result['format_valid'])

    def test_verify_gst_number_format(self):
        """Test verification of GST number format"""
        valid_gst = ['07AABCU9603R1ZV', '29GGGGG1314R9Z6', '27AAPFU0939F1ZV']
        invalid_gst = ['INVALID123', '07AABCU9603R1Z', 'ABCDEFGHIJKLMNO', '']
        
        for gst_num in valid_gst:
            with self.subTest(gst_number=gst_num):
                result = self.verifier.verify_gst_number_format(gst_num)
                self.assertTrue(result['format_valid'])
        
        for gst_num in invalid_gst:
            with self.subTest(gst_number=gst_num):
                result = self.verifier.verify_gst_number_format(gst_num)
                self.assertFalse(result['format_valid'])

    def test_cross_validate_extracted_data(self):
        """Test cross-validation of extracted data for consistency"""
        extracted_data = {
            'fields': self.mock_extracted_fields,
            'line_items': self.mock_line_items,
            'totals': self.mock_totals
        }
        
        cross_validation_result = self.verifier.cross_validate_extracted_data(extracted_data)
        
        self.assertIn('consistency_checks', cross_validation_result)
        self.assertIn('data_integrity_score', cross_validation_result)

    def test_decimal_precision_handling(self):
        """Test proper handling of decimal precision in calculations"""
        # Test with floating point precision issues
        line_items = [
            {
                'quantity': 3,
                'unit_price': 33.33,
                'total_amount': 99.99
            }
        ]
        
        verification_result = self.verifier.verify_line_item_calculations(line_items)
        
        # Should handle decimal precision correctly
        self.assertTrue(verification_result[0]['line_total_check']['check_passed'])

    def test_tolerance_based_verification(self):
        """Test verification with tolerance for small differences"""
        # Slight difference due to rounding
        line_items = [
            {
                'quantity': 3,
                'unit_price': 33.33,
                'total_amount': 100.00  # Slight difference from 99.99
            }
        ]
        
        verification_result = self.verifier.verify_line_item_calculations(
            line_items, tolerance=0.05
        )
        
        # Should pass with tolerance
        self.assertTrue(verification_result[0]['line_total_check']['check_passed'])

    def test_generate_verification_summary(self):
        """Test generation of comprehensive verification summary"""
        all_verification_data = {
            'field_verification': self.mock_extracted_fields,
            'line_items_verification': self.mock_line_items,
            'calculation_checks': {
                'subtotal_check': {'check_passed': True},
                'final_total_check': {'check_passed': True}
            }
        }
        
        summary = self.verifier.generate_verification_summary(all_verification_data)
        
        self.assertIn('overall_confidence', summary)
        self.assertIn('verification_passed', summary)
        self.assertIn('issues_found', summary)
        self.assertIn('recommendations', summary)

    def test_error_handling_empty_data(self):
        """Test error handling for empty or None data"""
        # Test with empty line items
        result = self.verifier.verify_line_item_calculations([])
        self.assertEqual(len(result), 0)
        
        # Test with None input
        result = self.verifier.verify_subtotal_calculation(None, 0)
        self.assertFalse(result['check_passed'])

    def test_error_handling_invalid_data_types(self):
        """Test error handling for invalid data types"""
        invalid_line_items = [
            {
                'quantity': 'invalid',  # Should be numeric
                'unit_price': 100.00,
                'total_amount': 'invalid'  # Should be numeric
            }
        ]
        
        with self.assertRaises((ValueError, TypeError)):
            self.verifier.verify_line_item_calculations(invalid_line_items)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVerifier)
    
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