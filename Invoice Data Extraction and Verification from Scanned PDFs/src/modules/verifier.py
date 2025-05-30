"""
Module 7: Verifier
Purpose: Validate extracted data and perform mathematical checks with confidence scoring
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
import math

class DataVerifier:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data verifier with configuration parameters
        
        Args:
            config: Configuration dictionary with verification parameters
        """
        self.config = config or {}
        self.tolerance = self.config.get('calculation_tolerance', 0.01)
        self.gst_rates = self.config.get('gst_rates', [0.0, 0.25, 5.0, 12.0, 18.0, 28.0])
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.5)
        self.logger = logging.getLogger(__name__)
        
        # Validation patterns
        self.patterns = {
            'invoice_number': r'^[A-Z0-9\-\/]{3,20}$',
            'gst_number': r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$',
            'pan_number': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$',
            'pincode': r'^[1-9][0-9]{5}$',
            'phone': r'^[6-9]\d{9}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        }
    
    def verify_line_item_calculations(self, line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify mathematical calculations for each line item
        
        Args:
            line_items: List of line item dictionaries
            
        Returns:
            Dictionary with verification results for line items
        """
        verification_result = {
            'is_valid': True,
            'total_items': len(line_items),
            'valid_items': 0,
            'invalid_items': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            for i, item in enumerate(line_items):
                item_result = self._verify_single_line_item(item, i)
                
                if item_result['is_valid']:
                    verification_result['valid_items'] += 1
                else:
                    verification_result['invalid_items'].append({
                        'item_index': i,
                        'errors': item_result['errors'],
                        'item_data': item
                    })
                    verification_result['errors'].extend(item_result['errors'])
                
                verification_result['warnings'].extend(item_result['warnings'])
            
            # Overall validation status
            if verification_result['invalid_items']:
                verification_result['is_valid'] = False
            
            self.logger.info(f"Line item verification: {verification_result['valid_items']}/{verification_result['total_items']} valid")
            
        except Exception as e:
            self.logger.error(f"Error in line item verification: {str(e)}")
            verification_result['is_valid'] = False
            verification_result['errors'].append(f"Verification error: {str(e)}")
        
        return verification_result
    
    def _verify_single_line_item(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Verify calculations for a single line item
        
        Args:
            item: Line item data
            index: Item index for error reporting
            
        Returns:
            Verification result for the item
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Extract numerical values with error handling
            quantity = self._safe_decimal_conversion(item.get('quantity', 0))
            unit_price = self._safe_decimal_conversion(item.get('unit_price', 0))
            total_amount = self._safe_decimal_conversion(item.get('total_amount', 0))
            discount = self._safe_decimal_conversion(item.get('discount', 0))
            
            # Calculate expected total
            if quantity is not None and unit_price is not None:
                expected_total = quantity * unit_price
                
                # Apply discount if present
                if discount > 0:
                    if item.get('discount_type') == 'percentage':
                        expected_total = expected_total * (1 - discount / 100)
                    else:
                        expected_total = expected_total - discount
                
                # Check if calculated total matches extracted total
                if total_amount is not None:
                    difference = abs(float(expected_total - total_amount))
                    
                    if difference > self.tolerance:
                        result['is_valid'] = False
                        result['errors'].append(
                            f"Item {index + 1}: Total amount mismatch. "
                            f"Expected: {expected_total}, Found: {total_amount}, "
                            f"Difference: {difference}"
                        )
                    elif difference > self.tolerance / 2:
                        result['warnings'].append(
                            f"Item {index + 1}: Minor calculation difference of {difference}"
                        )
                else:
                    result['warnings'].append(f"Item {index + 1}: Total amount not extracted")
            else:
                result['warnings'].append(f"Item {index + 1}: Missing quantity or unit price")
            
            # Validate HSN/SAC code if present
            hsn_code = item.get('hsn_sac_code')
            if hsn_code:
                if not self._validate_hsn_code(hsn_code):
                    result['warnings'].append(f"Item {index + 1}: Invalid HSN/SAC code format: {hsn_code}")
            
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Item {index + 1}: Calculation error - {str(e)}")
        
        return result
    
    def verify_subtotal_calculation(self, line_items: List[Dict[str, Any]], 
                                  extracted_subtotal: float) -> Dict[str, Any]:
        """
        Verify that subtotal matches sum of line item totals
        
        Args:
            line_items: List of line item dictionaries
            extracted_subtotal: Extracted subtotal value
            
        Returns:
            Dictionary with subtotal verification results
        """
        verification_result = {
            'is_valid': True,
            'calculated_subtotal': 0.0,
            'extracted_subtotal': extracted_subtotal,
            'difference': 0.0,
            'errors': [],
            'warnings': []
        }
        
        try:
            calculated_subtotal = Decimal('0.0')
            valid_items = 0
            
            for i, item in enumerate(line_items):
                total_amount = self._safe_decimal_conversion(item.get('total_amount', 0))
                
                if total_amount is not None:
                    calculated_subtotal += total_amount
                    valid_items += 1
                else:
                    verification_result['warnings'].append(
                        f"Line item {i + 1}: Total amount missing or invalid"
                    )
            
            verification_result['calculated_subtotal'] = float(calculated_subtotal)
            
            # Compare with extracted subtotal
            extracted_decimal = self._safe_decimal_conversion(extracted_subtotal)
            if extracted_decimal is not None:
                difference = abs(float(calculated_subtotal - extracted_decimal))
                verification_result['difference'] = difference
                
                if difference > self.tolerance:
                    verification_result['is_valid'] = False
                    verification_result['errors'].append(
                        f"Subtotal mismatch: Calculated: {calculated_subtotal}, "
                        f"Extracted: {extracted_subtotal}, Difference: {difference}"
                    )
                elif difference > self.tolerance / 2:
                    verification_result['warnings'].append(
                        f"Minor subtotal difference: {difference}"
                    )
            else:
                verification_result['warnings'].append("Extracted subtotal is invalid or missing")
            
            self.logger.info(f"Subtotal verification: {valid_items} items processed")
            
        except Exception as e:
            self.logger.error(f"Error in subtotal verification: {str(e)}")
            verification_result['is_valid'] = False
            verification_result['errors'].append(f"Subtotal verification error: {str(e)}")
        
        return verification_result
    
    def verify_final_total(self, subtotal: float, discount: float, 
                          gst_amount: float, final_total: float,
                          other_charges: float = 0.0) -> Dict[str, Any]:
        """
        Verify the final total calculation including taxes and discounts
        
        Args:
            subtotal: Subtotal amount
            discount: Discount amount
            gst_amount: GST amount
            final_total: Final total amount
            other_charges: Other charges (shipping, etc.)
            
        Returns:
            Dictionary with final total verification results
        """
        verification_result = {
            'is_valid': True,
            'calculated_total': 0.0,
            'extracted_total': final_total,
            'difference': 0.0,
            'breakdown': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Convert all values to Decimal for precise calculation
            subtotal_dec = self._safe_decimal_conversion(subtotal) or Decimal('0.0')
            discount_dec = self._safe_decimal_conversion(discount) or Decimal('0.0')
            gst_dec = self._safe_decimal_conversion(gst_amount) or Decimal('0.0')
            other_charges_dec = self._safe_decimal_conversion(other_charges) or Decimal('0.0')
            final_total_dec = self._safe_decimal_conversion(final_total)
            
            # Calculate expected final total
            calculated_total = subtotal_dec - discount_dec + gst_dec + other_charges_dec
            
            verification_result['calculated_total'] = float(calculated_total)
            verification_result['breakdown'] = {
                'subtotal': float(subtotal_dec),
                'discount': float(discount_dec),
                'gst_amount': float(gst_dec),
                'other_charges': float(other_charges_dec),
                'calculated_final_total': float(calculated_total)
            }
            
            # Compare with extracted final total
            if final_total_dec is not None:
                difference = abs(float(calculated_total - final_total_dec))
                verification_result['difference'] = difference
                
                if difference > self.tolerance:
                    verification_result['is_valid'] = False
                    verification_result['errors'].append(
                        f"Final total mismatch: Calculated: {calculated_total}, "
                        f"Extracted: {final_total}, Difference: {difference}"
                    )
                elif difference > self.tolerance / 2:
                    verification_result['warnings'].append(
                        f"Minor final total difference: {difference}"
                    )
            else:
                verification_result['warnings'].append("Extracted final total is invalid or missing")
            
            self.logger.info("Final total verification completed")
            
        except Exception as e:
            self.logger.error(f"Error in final total verification: {str(e)}")
            verification_result['is_valid'] = False
            verification_result['errors'].append(f"Final total verification error: {str(e)}")
        
        return verification_result
    
    def verify_gst_calculations(self, line_items: List[Dict[str, Any]], 
                               extracted_gst: Dict[str, float]) -> Dict[str, Any]:
        """
        Verify GST calculations for accuracy
        
        Args:
            line_items: List of line item dictionaries
            extracted_gst: Dictionary with GST breakdown (cgst, sgst, igst, etc.)
            
        Returns:
            Dictionary with GST verification results
        """
        verification_result = {
            'is_valid': True,
            'calculated_gst': {},
            'extracted_gst': extracted_gst,
            'gst_breakdown': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Calculate GST from line items
            gst_breakdown = {}
            total_taxable_amount = Decimal('0.0')
            
            for i, item in enumerate(line_items):
                gst_rate = self._safe_decimal_conversion(item.get('gst_rate', 0))
                taxable_amount = self._safe_decimal_conversion(item.get('taxable_amount', 
                                                             item.get('total_amount', 0)))
                
                if gst_rate is not None and taxable_amount is not None:
                    gst_amount = taxable_amount * gst_rate / 100
                    total_taxable_amount += taxable_amount
                    
                    rate_key = f"{float(gst_rate)}%"
                    if rate_key not in gst_breakdown:
                        gst_breakdown[rate_key] = {
                            'taxable_amount': Decimal('0.0'),
                            'gst_amount': Decimal('0.0'),
                            'items': []
                        }
                    
                    gst_breakdown[rate_key]['taxable_amount'] += taxable_amount
                    gst_breakdown[rate_key]['gst_amount'] += gst_amount
                    gst_breakdown[rate_key]['items'].append(i + 1)
            
            # Convert to float for output
            for rate_key in gst_breakdown:
                gst_breakdown[rate_key]['taxable_amount'] = float(gst_breakdown[rate_key]['taxable_amount'])
                gst_breakdown[rate_key]['gst_amount'] = float(gst_breakdown[rate_key]['gst_amount'])
            
            verification_result['gst_breakdown'] = gst_breakdown
            
            # Calculate total GST
            total_calculated_gst = sum(item['gst_amount'] for item in gst_breakdown.values())
            verification_result['calculated_gst']['total'] = total_calculated_gst
            
            # Compare with extracted GST
            extracted_total_gst = sum(extracted_gst.values()) if extracted_gst else 0
            
            if abs(total_calculated_gst - extracted_total_gst) > self.tolerance:
                verification_result['is_valid'] = False
                verification_result['errors'].append(
                    f"GST total mismatch: Calculated: {total_calculated_gst}, "
                    f"Extracted: {extracted_total_gst}"
                )
            
            self.logger.info(f"GST verification completed. Total GST: {total_calculated_gst}")
            
        except Exception as e:
            self.logger.error(f"Error in GST verification: {str(e)}")
            verification_result['is_valid'] = False
            verification_result['errors'].append(f"GST verification error: {str(e)}")
        
        return verification_result
    
    def validate_field_formats(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate format of extracted fields using regex patterns
        
        Args:
            extracted_data: Dictionary with extracted invoice data
            
        Returns:
            Dictionary with format validation results
        """
        validation_result = {
            'is_valid': True,
            'field_validations': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate each field with its pattern
            for field_name, pattern in self.patterns.items():
                field_value = extracted_data.get(field_name)
                
                if field_value:
                    field_result = {
                        'is_valid': bool(re.match(pattern, str(field_value).strip())),
                        'value': field_value,
                        'pattern': pattern
                    }
                    
                    if not field_result['is_valid']:
                        validation_result['is_valid'] = False
                        validation_result['errors'].append(
                            f"Invalid format for {field_name}: {field_value}"
                        )
                    
                    validation_result['field_validations'][field_name] = field_result
            
            # Additional custom validations
            self._validate_dates(extracted_data, validation_result)
            self._validate_amounts(extracted_data, validation_result)
            
            self.logger.info(f"Format validation completed. Valid: {validation_result['is_valid']}")
            
        except Exception as e:
            self.logger.error(f"Error in format validation: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Format validation error: {str(e)}")
        
        return validation_result
    
    def _validate_dates(self, extracted_data: Dict[str, Any], validation_result: Dict[str, Any]):
        """
        Validate date fields for logical consistency
        
        Args:
            extracted_data: Extracted invoice data
            validation_result: Validation result to update
        """
        try:
            invoice_date = extracted_data.get('invoice_date')
            due_date = extracted_data.get('due_date')
            
            if invoice_date and due_date:
                try:
                    inv_date = datetime.strptime(str(invoice_date), '%Y-%m-%d')
                    d_date = datetime.strptime(str(due_date), '%Y-%m-%d')
                    
                    if d_date < inv_date:
                        validation_result['warnings'].append(
                            "Due date is before invoice date"
                        )
                    
                    # Check if dates are reasonable (not too far in past/future)
                    current_date = datetime.now()
                    if inv_date > current_date + timedelta(days=30):
                        validation_result['warnings'].append(
                            "Invoice date is in the future"
                        )
                    
                    if inv_date < current_date - timedelta(days=3650):  # 10 years
                        validation_result['warnings'].append(
                            "Invoice date is very old"
                        )
                        
                except ValueError:
                    validation_result['errors'].append(
                        "Invalid date format in invoice_date or due_date"
                    )
                    validation_result['is_valid'] = False
                    
        except Exception as e:
            validation_result['warnings'].append(f"Date validation error: {str(e)}")
    
    def _validate_amounts(self, extracted_data: Dict[str, Any], validation_result: Dict[str, Any]):
        """
        Validate amount fields for logical consistency
        
        Args:
            extracted_data: Extracted invoice data
            validation_result: Validation result to update
        """
        try:
            amounts = {
                'subtotal': extracted_data.get('subtotal', 0),
                'total_amount': extracted_data.get('total_amount', 0),
                'gst_amount': extracted_data.get('total_gst', 0),
                'discount': extracted_data.get('total_discount', 0)
            }
            
            # Check for negative amounts where they shouldn't be
            for field, value in amounts.items():
                if field != 'discount' and value and float(value) < 0:
                    validation_result['warnings'].append(
                        f"Negative amount found in {field}: {value}"
                    )
            
            # Check logical relationships
            subtotal = float(amounts['subtotal'] or 0)
            total = float(amounts['total_amount'] or 0)
            
            if subtotal > 0 and total > 0 and total < subtotal * 0.5:
                validation_result['warnings'].append(
                    "Total amount is significantly less than subtotal"
                )
                
        except Exception as e:
            validation_result['warnings'].append(f"Amount validation error: {str(e)}")
    
    def generate_verification_flags(self, all_checks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overall verification flags based on all validation checks
        
        Args:
            all_checks: Dictionary containing results from all verification checks
            
        Returns:
            Dictionary with overall verification flags and status
        """
        verification_flags = {
            'overall_status': 'valid',
            'confidence_score': 1.0,
            'critical_errors': [],
            'warnings': [],
            'verification_summary': {
                'line_items_valid': False,
                'calculations_valid': False,
                'formats_valid': False,
                'gst_valid': False
            },
            'recommendation': 'approved'
        }
        
        try:
            critical_error_count = 0
            warning_count = 0
            
            # Process each check result
            for check_name, check_result in all_checks.items():
                if not check_result.get('is_valid', True):
                    critical_error_count += len(check_result.get('errors', []))
                    verification_flags['critical_errors'].extend(
                        [f"{check_name}: {error}" for error in check_result.get('errors', [])]
                    )
                
                warning_count += len(check_result.get('warnings', []))
                verification_flags['warnings'].extend(
                    [f"{check_name}: {warning}" for warning in check_result.get('warnings', [])]
                )
                
                # Update verification summary
                if check_name == 'line_items':
                    verification_flags['verification_summary']['line_items_valid'] = check_result.get('is_valid', False)
                elif check_name in ['subtotal', 'final_total']:
                    verification_flags['verification_summary']['calculations_valid'] = check_result.get('is_valid', False)
                elif check_name == 'formats':
                    verification_flags['verification_summary']['formats_valid'] = check_result.get('is_valid', False)
                elif check_name == 'gst':
                    verification_flags['verification_summary']['gst_valid'] = check_result.get('is_valid', False)
            
            # Calculate confidence score
            total_checks = len(all_checks)
            if total_checks > 0:
                # Base confidence on successful validations
                successful_checks = sum(1 for check in all_checks.values() if check.get('is_valid', False))
                base_confidence = successful_checks / total_checks
                
                # Reduce confidence based on errors and warnings
                error_penalty = min(critical_error_count * 0.1, 0.5)
                warning_penalty = min(warning_count * 0.02, 0.2)
                
                verification_flags['confidence_score'] = max(0.0, base_confidence - error_penalty - warning_penalty)
            
            # Determine overall status and recommendation
            if critical_error_count == 0:
                if warning_count == 0:
                    verification_flags['overall_status'] = 'valid'
                    verification_flags['recommendation'] = 'approved'
                elif warning_count <= 3:
                    verification_flags['overall_status'] = 'valid_with_warnings'
                    verification_flags['recommendation'] = 'approved_with_review'
                else:
                    verification_flags['overall_status'] = 'questionable'
                    verification_flags['recommendation'] = 'manual_review_required'
            else:
                if critical_error_count <= 2:
                    verification_flags['overall_status'] = 'invalid_minor'
                    verification_flags['recommendation'] = 'manual_review_required'
                else:
                    verification_flags['overall_status'] = 'invalid_major'
                    verification_flags['recommendation'] = 'rejected'
            
            self.logger.info(f"Verification flags generated. Status: {verification_flags['overall_status']}, "
                           f"Confidence: {verification_flags['confidence_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error generating verification flags: {str(e)}")
            verification_flags['overall_status'] = 'error'
            verification_flags['recommendation'] = 'manual_review_required'
            verification_flags['critical_errors'].append(f"Verification error: {str(e)}")
        
        return verification_flags
    
    def create_confidence_report(self, field_confidences: Dict[str, float], 
                               ocr_confidences: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Create a comprehensive confidence report for all extracted fields
        
        Args:
            field_confidences: Confidence scores for extracted fields
            ocr_confidences: Raw OCR confidence scores
            
        Returns:
            Dictionary with detailed confidence analysis
        """
        confidence_report = {
            'overall_confidence': 0.0,
            'field_confidence_breakdown': {},
            'confidence_categories': {
                'high_confidence': [],    # > 0.8
                'medium_confidence': [],  # 0.5 - 0.8
                'low_confidence': []      # < 0.5
            },
            'recommendations': [],
            'quality_metrics': {}
        }
        
        try:
            if not field_confidences:
                confidence_report['recommendations'].append("No confidence data available")
                return confidence_report
            
            # Process field confidences
            total_confidence = 0.0
            field_count = 0
            
            for field_name, confidence in field_confidences.items():
                field_confidence_data = {
                    'confidence': confidence,
                    'category': self._categorize_confidence(confidence),
                    'ocr_confidence': ocr_confidences.get(field_name) if ocr_confidences else None
                }
                
                confidence_report['field_confidence_breakdown'][field_name] = field_confidence_data
                
                # Categorize field
                if confidence > 0.8:
                    confidence_report['confidence_categories']['high_confidence'].append(field_name)
                elif confidence > 0.5:
                    confidence_report['confidence_categories']['medium_confidence'].append(field_name)
                else:
                    confidence_report['confidence_categories']['low_confidence'].append(field_name)
                
                total_confidence += confidence
                field_count += 1
            
            # Calculate overall confidence
            if field_count > 0:
                confidence_report['overall_confidence'] = total_confidence / field_count
            
            # Generate quality metrics
            confidence_report['quality_metrics'] = {
                'total_fields': field_count,
                'high_confidence_count': len(confidence_report['confidence_categories']['high_confidence']),
                'medium_confidence_count': len(confidence_report['confidence_categories']['medium_confidence']),
                'low_confidence_count': len(confidence_report['confidence_categories']['low_confidence']),
                'high_confidence_percentage': len(confidence_report['confidence_categories']['high_confidence']) / field_count * 100 if field_count > 0 else 0
            }
            
            # Generate recommendations
            self._generate_confidence_recommendations(confidence_report)
            
            self.logger.info(f"Confidence report created. Overall confidence: {confidence_report['overall_confidence']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error creating confidence report: {str(e)}")
            confidence_report['recommendations'].append(f"Error in confidence analysis: {str(e)}")
        
        return confidence_report
    
    def _safe_decimal_conversion(self, value: Any) -> Optional[Decimal]:
        """
        Safely convert a value to Decimal, handling various input types
        
        Args:
            value: Value to convert
            
        Returns:
            Decimal value or None if conversion fails
        """
        if value is None:
            return None
        
        try:
            # Handle string values
            if isinstance(value, str):
                # Remove common formatting characters
                cleaned_value = re.sub(r'[₹,$\s,]', '', value.strip())
                if not cleaned_value:
                    return None
                return Decimal(cleaned_value)
            
            # Handle numeric values
            return Decimal(str(value))
            
        except (InvalidOperation, ValueError, TypeError):
            return None
    
    def _validate_hsn_code(self, hsn_code: str) -> bool:
        """
        Validate HSN/SAC code format
        
        Args:
            hsn_code: HSN/SAC code to validate
            
        Returns:
            True if valid format, False otherwise
        """
        if not hsn_code:
            return False
        
        # HSN codes are typically 4, 6, or 8 digits
        # SAC codes are typically 6 digits
        hsn_pattern = r'^[0-9]{4}([0-9]{2}([0-9]{2})?)?$'
        return bool(re.match(hsn_pattern, str(hsn_code).strip()))
    
    def _categorize_confidence(self, confidence: float) -> str:
        """
        Categorize confidence score into descriptive categories
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Confidence category string
        """
        if confidence > 0.8:
            return 'high'
        elif confidence > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_confidence_recommendations(self, confidence_report: Dict[str, Any]):
        """
        Generate recommendations based on confidence analysis
        
        Args:
            confidence_report: Confidence report to update with recommendations
        """
        recommendations = []
        
        # Overall confidence recommendations
        overall_conf = confidence_report['overall_confidence']
        if overall_conf < 0.5:
            recommendations.append("Overall confidence is low. Manual review strongly recommended.")
        elif overall_conf < 0.7:
            recommendations.append("Overall confidence is moderate. Review critical fields manually.")
        
        # Field-specific recommendations
        low_conf_fields = confidence_report['confidence_categories']['low_confidence']
        if low_conf_fields:
            recommendations.append(f"Low confidence fields requiring attention: {', '.join(low_conf_fields)}")
        
        # Quality-based recommendations
        metrics = confidence_report['quality_metrics']
        if metrics['high_confidence_percentage'] < 50:
            recommendations.append("Less than 50% of fields have high confidence. Consider re-processing with better image quality.")
        
        confidence_report['recommendations'] = recommendations


def main():
    """
    Test function for the verifier module
    """
    # Test data
    test_line_items = [
        {
            'quantity': 2,
            'unit_price': 100.0,
            'total_amount': 200.0,
            'gst_rate': 18,
            'hsn_sac_code': '1234'
        },
        {
            'quantity': 1,
            'unit_price': 500.0,
            'total_amount': 500.0,
            'gst_rate': 18,
            'hsn_sac_code': '5678'
        }
    ]
    
    test_confidences = {
        'invoice_number': 0.9,
        'invoice_date': 0.8,
        'total_amount': 0.7,
        'gst_number': 0.6
    }
    
    # Test the verifier
    verifier = DataVerifier()
    
    # Test line item verification
    line_item_result = verifier.verify_line_item_calculations(test_line_items)
    print("Line Item Verification:", line_item_result)
    
    # Test subtotal verification
    subtotal_result = verifier.verify_subtotal_calculation(test_line_items, 700.0)
    print("Subtotal Verification:", subtotal_result)
    
    # Test confidence report
    confidence_report = verifier.create_confidence_report(test_confidences)
    print("Confidence Report:", confidence_report)


if __name__ == "__main__":
    main()