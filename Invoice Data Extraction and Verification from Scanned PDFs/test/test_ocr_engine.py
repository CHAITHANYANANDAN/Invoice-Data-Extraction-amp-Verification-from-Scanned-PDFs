"""
Test module for OCR Engine functionality
Tests OCR text extraction, confidence scoring, and coordinate detection
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modules.ocr_engine import OCREngine


class TestOCREngine(unittest.TestCase):
    """Test cases for OCR Engine module"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.ocr_engine = OCREngine()
        
        # Mock image data (simulating a grayscale image)
        self.mock_image = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        
        # Mock OCR results structure
        self.mock_tesseract_data = {
            'text': ['INVOICE', 'NUMBER:', '12345', 'DATE:', '2024-05-15'],
            'left': [10, 80, 150, 10, 80],
            'top': [20, 20, 20, 40, 40],
            'width': [60, 70, 45, 40, 85],
            'height': [15, 15, 15, 15, 15],
            'conf': [95, 88, 92, 90, 85]
        }
        
        self.mock_easyocr_results = [
            ([[10, 20], [70, 20], [70, 35], [10, 35]], 'INVOICE', 0.95),
            ([[80, 20], [150, 20], [150, 35], [80, 35]], 'NUMBER:', 0.88),
            ([[150, 20], [195, 20], [195, 35], [150, 35]], '12345', 0.92),
            ([[10, 40], [50, 40], [50, 55], [10, 55]], 'DATE:', 0.90),
            ([[80, 40], [165, 40], [165, 55], [80, 55]], '2024-05-15', 0.85)
        ]

    def test_initialize_ocr_model_tesseract(self):
        """Test OCR model initialization with Tesseract"""
        with patch('pytesseract.get_tesseract_version') as mock_version:
            mock_version.return_value = "5.0.0"
            
            ocr_engine = OCREngine(engine='tesseract')
            self.assertEqual(ocr_engine.engine, 'tesseract')
            self.assertIsNotNone(ocr_engine.tesseract_config)

    def test_initialize_ocr_model_easyocr(self):
        """Test OCR model initialization with EasyOCR"""
        with patch('easyocr.Reader') as mock_reader:
            mock_reader.return_value = MagicMock()
            
            ocr_engine = OCREngine(engine='easyocr')
            self.assertEqual(ocr_engine.engine, 'easyocr')
            self.assertIsNotNone(ocr_engine.easyocr_reader)

    def test_initialize_ocr_model_invalid_engine(self):
        """Test OCR model initialization with invalid engine"""
        with self.assertRaises(ValueError):
            OCREngine(engine='invalid_engine')

    @patch('pytesseract.image_to_data')
    def test_extract_text_with_coordinates_tesseract(self, mock_image_to_data):
        """Test text extraction with coordinates using Tesseract"""
        mock_image_to_data.return_value = self.mock_tesseract_data
        
        ocr_engine = OCREngine(engine='tesseract')
        results = ocr_engine.extract_text_with_coordinates(self.mock_image)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)
        
        # Check first result structure
        first_result = results[0]
        self.assertIn('text', first_result)
        self.assertIn('bbox', first_result)
        self.assertIn('confidence', first_result)
        self.assertEqual(first_result['text'], 'INVOICE')
        self.assertEqual(first_result['confidence'], 95)

    @patch('easyocr.Reader')
    def test_extract_text_with_coordinates_easyocr(self, mock_reader_class):
        """Test text extraction with coordinates using EasyOCR"""
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = self.mock_easyocr_results
        mock_reader_class.return_value = mock_reader
        
        ocr_engine = OCREngine(engine='easyocr')
        results = ocr_engine.extract_text_with_coordinates(self.mock_image)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)
        
        # Check first result structure
        first_result = results[0]
        self.assertIn('text', first_result)
        self.assertIn('bbox', first_result)
        self.assertIn('confidence', first_result)
        self.assertEqual(first_result['text'], 'INVOICE')
        self.assertEqual(first_result['confidence'], 0.95)

    def test_get_confidence_scores(self):
        """Test confidence score extraction from OCR results"""
        mock_results = [
            {'text': 'INVOICE', 'confidence': 95},
            {'text': 'NUMBER:', 'confidence': 88},
            {'text': '12345', 'confidence': 92}
        ]
        
        confidence_scores = self.ocr_engine.get_confidence_scores(mock_results)
        
        self.assertEqual(len(confidence_scores), 3)
        self.assertEqual(confidence_scores[0], 95)
        self.assertEqual(confidence_scores[1], 88)
        self.assertEqual(confidence_scores[2], 92)

    def test_filter_low_confidence_results(self):
        """Test filtering of low confidence OCR results"""
        mock_results = [
            {'text': 'INVOICE', 'confidence': 95},
            {'text': 'blurred_text', 'confidence': 30},
            {'text': '12345', 'confidence': 92},
            {'text': 'noise', 'confidence': 15}
        ]
        
        filtered_results = self.ocr_engine.filter_low_confidence_results(
            mock_results, threshold=50
        )
        
        self.assertEqual(len(filtered_results), 2)
        self.assertEqual(filtered_results[0]['text'], 'INVOICE')
        self.assertEqual(filtered_results[1]['text'], '12345')

    def test_merge_nearby_text_blocks(self):
        """Test merging of nearby text blocks"""
        mock_results = [
            {'text': 'INVOICE', 'bbox': (10, 20, 70, 35), 'confidence': 95},
            {'text': 'NUMBER:', 'bbox': (75, 20, 130, 35), 'confidence': 88},
            {'text': '12345', 'bbox': (135, 20, 180, 35), 'confidence': 92}
        ]
        
        merged_results = self.ocr_engine.merge_nearby_text_blocks(
            mock_results, distance_threshold=10
        )
        
        # Should merge first two blocks that are close
        self.assertLessEqual(len(merged_results), len(mock_results))

    def test_extract_full_text(self):
        """Test extraction of full text from image"""
        with patch.object(self.ocr_engine, 'extract_text_with_coordinates') as mock_extract:
            mock_extract.return_value = [
                {'text': 'INVOICE', 'confidence': 95},
                {'text': 'NUMBER:', 'confidence': 88},
                {'text': '12345', 'confidence': 92}
            ]
            
            full_text = self.ocr_engine.extract_full_text(self.mock_image)
            expected_text = "INVOICE NUMBER: 12345"
            
            self.assertEqual(full_text, expected_text)

    def test_get_text_regions_by_confidence(self):
        """Test grouping text regions by confidence levels"""
        mock_results = [
            {'text': 'HIGH_CONF', 'confidence': 95},
            {'text': 'MED_CONF', 'confidence': 75},
            {'text': 'LOW_CONF', 'confidence': 45},
            {'text': 'VERY_HIGH', 'confidence': 98}
        ]
        
        regions = self.ocr_engine.get_text_regions_by_confidence(mock_results)
        
        self.assertIn('high', regions)
        self.assertIn('medium', regions)
        self.assertIn('low', regions)
        
        # Check high confidence region
        high_conf_texts = [item['text'] for item in regions['high']]
        self.assertIn('HIGH_CONF', high_conf_texts)
        self.assertIn('VERY_HIGH', high_conf_texts)

    def test_extract_text_in_region(self):
        """Test text extraction within specific image region"""
        # Define a region of interest (x, y, width, height)
        roi = (50, 10, 100, 50)
        
        with patch.object(self.ocr_engine, 'extract_text_with_coordinates') as mock_extract:
            mock_extract.return_value = [
                {'text': 'INSIDE_ROI', 'bbox': (60, 20, 120, 35), 'confidence': 95},
                {'text': 'OUTSIDE_ROI', 'bbox': (200, 20, 250, 35), 'confidence': 88}
            ]
            
            roi_text = self.ocr_engine.extract_text_in_region(self.mock_image, roi)
            
            # Should only return text within the ROI
            self.assertEqual(len(roi_text), 1)
            self.assertEqual(roi_text[0]['text'], 'INSIDE_ROI')

    def test_preprocess_for_ocr(self):
        """Test image preprocessing for better OCR results"""
        processed_image = self.ocr_engine.preprocess_for_ocr(self.mock_image)
        
        # Should return processed image with same dimensions
        self.assertEqual(processed_image.shape, self.mock_image.shape)

    def test_validate_ocr_results(self):
        """Test validation of OCR results"""
        valid_results = [
            {'text': 'INVOICE', 'bbox': (10, 20, 70, 35), 'confidence': 95},
            {'text': '12345', 'bbox': (80, 20, 120, 35), 'confidence': 88}
        ]
        
        invalid_results = [
            {'text': '', 'bbox': (10, 20, 70, 35), 'confidence': 95},  # Empty text
            {'text': 'valid', 'confidence': 5}  # Missing bbox
        ]
        
        self.assertTrue(self.ocr_engine.validate_ocr_results(valid_results))
        self.assertFalse(self.ocr_engine.validate_ocr_results(invalid_results))

    def test_error_handling_invalid_image(self):
        """Test error handling for invalid image input"""
        with self.assertRaises(ValueError):
            self.ocr_engine.extract_text_with_coordinates(None)
        
        with self.assertRaises(ValueError):
            self.ocr_engine.extract_text_with_coordinates("not_an_image")

    def test_performance_metrics(self):
        """Test OCR performance metrics calculation"""
        with patch.object(self.ocr_engine, 'extract_text_with_coordinates') as mock_extract:
            import time
            
            # Mock processing time
            mock_extract.return_value = [
                {'text': 'TEST', 'confidence': 95}
            ]
            
            start_time = time.time()
            results = self.ocr_engine.extract_text_with_coordinates(self.mock_image)
            processing_time = time.time() - start_time
            
            metrics = self.ocr_engine.get_performance_metrics(
                results, processing_time
            )
            
            self.assertIn('processing_time', metrics)
            self.assertIn('text_blocks_detected', metrics)
            self.assertIn('average_confidence', metrics)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOCREngine)
    
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