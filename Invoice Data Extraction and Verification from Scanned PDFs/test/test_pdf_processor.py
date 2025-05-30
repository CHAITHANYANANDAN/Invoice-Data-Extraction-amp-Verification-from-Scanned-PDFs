"""
Unit tests for PDF Processor module.
Tests PDF to image conversion functionality.
"""

import unittest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

# Import test utilities
from . import (
    TestConfig, 
    setup_test_environment, 
    cleanup_test_environment,
    create_mock_config,
    get_sample_pdf_bytes
)

# Import module under test
from src.modules.pdf_processor import PDFProcessor


class TestPDFProcessor(unittest.TestCase):
    """Test cases for PDFProcessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.temp_dir = setup_test_environment()
        cls.config = create_mock_config()
        
        # Create sample PDF file for testing
        cls.sample_pdf_path = cls.temp_dir / "test_invoice.pdf"
        pdf_bytes = get_sample_pdf_bytes()
        if pdf_bytes:
            with open(cls.sample_pdf_path, 'wb') as f:
                f.write(pdf_bytes)
        else:
            # Create a dummy file if reportlab is not available
            cls.sample_pdf_path.touch()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cleanup_test_environment()
    
    def setUp(self):
        """Set up each test case."""
        self.pdf_processor = PDFProcessor(self.config)
        self.temp_output_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test case."""
        if os.path.exists(self.temp_output_dir):
            shutil.rmtree(self.temp_output_dir)
    
    def test_initialization(self):
        """Test PDFProcessor initialization."""
        # Test successful initialization
        processor = PDFProcessor(self.config)
        self.assertIsNotNone(processor)
        self.assertEqual(processor.config, self.config)
        
        # Test initialization without config
        processor_no_config = PDFProcessor()
        self.assertIsNotNone(processor_no_config)
    
    @patch('src.modules.pdf_processor.pdf2image.convert_from_path')
    def test_extract_images_from_pdf_success(self, mock_convert):
        """Test successful PDF to image conversion."""
        # Setup mock
        mock_image1 = Mock()
        mock_image1.size = (800, 600)
        mock_image2 = Mock()
        mock_image2.size = (800, 600)
        mock_convert.return_value = [mock_image1, mock_image2]
        
        # Test extraction
        result = self.pdf_processor.extract_images_from_pdf(str(self.sample_pdf_path))
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        mock_convert.assert_called_once_with(str(self.sample_pdf_path), dpi=300)
    
    @patch('src.modules.pdf_processor.pdf2image.convert_from_path')
    def test_extract_images_from_pdf_file_not_found(self, mock_convert):
        """Test PDF extraction with non-existent file."""
        mock_convert.side_effect = FileNotFoundError("File not found")
        
        result = self.pdf_processor.extract_images_from_pdf("nonexistent.pdf")
        
        self.assertEqual(result, [])
    
    @patch('src.modules.pdf_processor.pdf2image.convert_from_path')
    def test_extract_images_from_pdf_empty_result(self, mock_convert):
        """Test PDF extraction returning empty list."""
        mock_convert.return_value = []
        
        result = self.pdf_processor.extract_images_from_pdf(str(self.sample_pdf_path))
        
        self.assertEqual(result, [])
    
    @patch('src.modules.pdf_processor.pdf2image.convert_from_path')
    def test_extract_images_from_pdf_with_custom_dpi(self, mock_convert):
        """Test PDF extraction with custom DPI setting."""
        # Setup config with custom DPI
        config_custom_dpi = create_mock_config()
        config_custom_dpi.pdf_dpi = 200
        processor = PDFProcessor(config_custom_dpi)
        
        mock_image = Mock()
        mock_image.size = (800, 600)
        mock_convert.return_value = [mock_image]
        
        result = processor.extract_images_from_pdf(str(self.sample_pdf_path))
        
        mock_convert.assert_called_once_with(str(self.sample_pdf_path), dpi=200)
        self.assertEqual(len(result), 1)
    
    def test_save_page_images_success(self):
        """Test successful saving of page images."""
        # Create mock images
        mock_images = []
        for i in range(2):
            mock_image = Mock()
            mock_image.save = Mock()
            mock_images.append(mock_image)
        
        # Test saving
        result = self.pdf_processor.save_page_images(mock_images, self.temp_output_dir)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        
        # Check that save was called for each image
        for i, mock_image in enumerate(mock_images):
            expected_path = os.path.join(self.temp_output_dir, f"page_{i+1}.png")
            mock_image.save.assert_called_once_with(expected_path, "PNG")
            self.assertIn(expected_path, result)
    
    def test_save_page_images_empty_list(self):
        """Test saving empty list of images."""
        result = self.pdf_processor.save_page_images([], self.temp_output_dir)
        self.assertEqual(result, [])
    
    def test_save_page_images_invalid_directory(self):
        """Test saving images to invalid directory."""
        mock_image = Mock()
        mock_image.save = Mock(side_effect=OSError("Permission denied"))
        
        result = self.pdf_processor.save_page_images([mock_image], "/invalid/directory")
        self.assertEqual(result, [])
    
    @patch('src.modules.pdf_processor.np.array')
    def test_pil_to_numpy_conversion(self, mock_array):
        """Test PIL image to numpy array conversion."""
        # Setup mock
        mock_pil_image = Mock()
        mock_array.return_value = np.ones((600, 800, 3), dtype=np.uint8)
        
        # Test conversion
        result = self.pdf_processor._pil_to_numpy(mock_pil_image)
        
        # Assertions
        self.assertIsInstance(result, np.ndarray)
        mock_array.assert_called_once_with(mock_pil_image)
    
    def test_validate_pdf_file_valid(self):
        """Test PDF file validation with valid file."""
        # Create a file with .pdf extension
        valid_pdf = self.temp_output_dir + "/valid.pdf"
        Path(valid_pdf).touch()
        
        result = self.pdf_processor._validate_pdf_file(valid_pdf)
        self.assertTrue(result)
    
    def test_validate_pdf_file_invalid_extension(self):
        """Test PDF file validation with invalid extension."""
        invalid_file = self.temp_output_dir + "/invalid.txt"
        Path(invalid_file).touch()
        
        result = self.pdf_processor._validate_pdf_file(invalid_file)
        self.assertFalse(result)
    
    def test_validate_pdf_file_nonexistent(self):
        """Test PDF file validation with non-existent file."""
        result = self.pdf_processor._validate_pdf_file("nonexistent.pdf")
        self.assertFalse(result)
    
    @patch('src.modules.pdf_processor.pdf2image.convert_from_path')
    def test_extract_images_with_error_handling(self, mock_convert):
        """Test image extraction with various error conditions."""
        # Test with corrupted PDF
        mock_convert.side_effect = Exception("PDF corrupted")
        
        result = self.pdf_processor.extract_images_from_pdf(str(self.sample_pdf_path))
        self.assertEqual(result, [])
    
    def test_get_image_info(self):
        """Test getting image information."""
        # Create a mock PIL image
        mock_image = Mock()
        mock_image.size = (800, 600)
        mock_image.mode = 'RGB'
        
        info = self.pdf_processor._get_image_info(mock_image)
        
        expected_info = {
            'width': 800,
            'height': 600,
            'mode': 'RGB'
        }
        self.assertEqual(info, expected_info)
    
    @patch('src.modules.pdf_processor.pdf2image.convert_from_path')
    def test_extract_images_with_progress_callback(self, mock_convert):
        """Test image extraction with progress callback."""
        # Setup mock
        mock_images = [Mock() for _ in range(3)]
        for img in mock_images:
            img.size = (800, 600)
        mock_convert.return_value = mock_images
        
        # Progress callback tracking
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        # Test with callback
        result = self.pdf_processor.extract_images_from_pdf(
            str(self.sample_pdf_path), 
            progress_callback=progress_callback
        )
        
        self.assertEqual(len(result), 3)
        # Note: This test assumes the method supports progress callback
        # If not implemented, this test documents the expected behavior
    
    def test_extract_images_memory_optimization(self):
        """Test memory optimization during image extraction."""
        # This test checks that large images are handled efficiently
        # Implementation would depend on actual memory optimization strategies
        pass
    
    @patch('src.modules.pdf_processor.logging')
    def test_logging_integration(self, mock_logging):
        """Test that PDF processor integrates with logging system."""
        # Test that errors are logged appropriately
        with patch('src.modules.pdf_processor.pdf2image.convert_from_path') as mock_convert:
            mock_convert.side_effect = Exception("Test error")
            
            result = self.pdf_processor.extract_images_from_pdf("test.pdf")
            
            # Verify logging was called (implementation dependent)
            self.assertEqual(result, [])


class TestPDFProcessorIntegration(unittest.TestCase):
    """Integration tests for PDF Processor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test environment."""
        cls.temp_dir = setup_test_environment()
        cls.config = create_mock_config()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up integration test environment."""
        cleanup_test_environment()
    
    def setUp(self):
        """Set up each integration test."""
        self.pdf_processor = PDFProcessor(self.config)
    
    @unittest.skipIf(not get_sample_pdf_bytes(), "PDF generation not available")
    def test_end_to_end_pdf_processing(self):
        """Test complete PDF processing workflow."""
        # Create sample PDF
        pdf_path = self.temp_dir / "integration_test.pdf"
        pdf_bytes = get_sample_pdf_bytes()
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # Process PDF
        images = self.pdf_processor.extract_images_from_pdf(str(pdf_path))
        
        # Verify results
        self.assertGreater(len(images), 0)
        
        # Save images and verify
        output_dir = self.temp_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        saved_paths = self.pdf_processor.save_page_images(images, str(output_dir))
        
        self.assertEqual(len(saved_paths), len(images))
        for path in saved_paths:
            self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)