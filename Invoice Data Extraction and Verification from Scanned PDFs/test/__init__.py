"""
Test package for Invoice Data Extraction Project.
Contains unit tests for all processing modules.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path for imports
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test configuration and utilities
class TestConfig:
    """Test configuration constants and settings."""
    
    # Test data paths
    TEST_DATA_DIR = TEST_DIR / "test_data"
    SAMPLE_PDF_PATH = TEST_DATA_DIR / "sample_invoice.pdf"
    SAMPLE_IMAGE_PATH = TEST_DATA_DIR / "sample_invoice_page.png"
    
    # Temporary directories for test outputs
    TEMP_OUTPUT_DIR = None
    
    # Test thresholds and tolerances
    OCR_CONFIDENCE_THRESHOLD = 0.5
    CALCULATION_TOLERANCE = 0.01
    
    # Sample test data
    SAMPLE_INVOICE_DATA = {
        'invoice_number': 'INV-2024-001',
        'invoice_date': '2024-01-15',
        'supplier_gst': '29ABCDE1234F1Z5',
        'buyer_gst': '27FGHIJ5678K2L9',
        'po_number': 'PO-2024-0001',
        'line_items': [
            {
                'description': 'Product A',
                'hsn_code': '1234',
                'quantity': 10,
                'unit_price': 100.0,
                'total_amount': 1000.0
            },
            {
                'description': 'Product B',
                'hsn_code': '5678',
                'quantity': 5,
                'unit_price': 200.0,
                'total_amount': 1000.0
            }
        ],
        'subtotal': 2000.0,
        'gst_amount': 360.0,
        'final_total': 2360.0
    }
    
    # Sample OCR results for testing
    SAMPLE_OCR_RESULT = [
        {
            'text': 'INVOICE',
            'confidence': 95.5,
            'bbox': [100, 50, 200, 80]
        },
        {
            'text': 'Invoice Number: INV-2024-001',
            'confidence': 92.3,
            'bbox': [100, 100, 300, 120]
        },
        {
            'text': 'Date: 15/01/2024',
            'confidence': 88.7,
            'bbox': [100, 130, 250, 150]
        },
        {
            'text': 'GST: 29ABCDE1234F1Z5',
            'confidence': 90.1,
            'bbox': [100, 160, 280, 180]
        }
    ]

def setup_test_environment():
    """Setup test environment with temporary directories and sample data."""
    # Create test data directory
    TestConfig.TEST_DATA_DIR.mkdir(exist_ok=True)
    
    # Create temporary output directory
    TestConfig.TEMP_OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="invoice_test_"))
    
    # Create sample test files if they don't exist
    create_sample_test_files()
    
    return TestConfig.TEMP_OUTPUT_DIR

def cleanup_test_environment():
    """Clean up test environment and temporary files."""
    if TestConfig.TEMP_OUTPUT_DIR and TestConfig.TEMP_OUTPUT_DIR.exists():
        shutil.rmtree(TestConfig.TEMP_OUTPUT_DIR)

def create_sample_test_files():
    """Create sample test files for testing."""
    import numpy as np
    from PIL import Image
    
    # Create a simple test image if it doesn't exist
    if not TestConfig.SAMPLE_IMAGE_PATH.exists():
        # Create a simple white image with some text-like patterns
        image_array = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Add some black rectangles to simulate text
        image_array[100:120, 100:300] = 0  # Header line
        image_array[150:170, 100:250] = 0  # Invoice number
        image_array[200:220, 100:200] = 0  # Date
        image_array[300:500, 100:500] = 0  # Table area
        
        image = Image.fromarray(image_array)
        image.save(TestConfig.SAMPLE_IMAGE_PATH)

def get_sample_pdf_bytes():
    """Generate sample PDF bytes for testing."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        import io
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Add sample invoice content
        p.drawString(100, 750, "INVOICE")
        p.drawString(100, 700, "Invoice Number: INV-2024-001")
        p.drawString(100, 650, "Date: 15/01/2024")
        p.drawString(100, 600, "GST: 29ABCDE1234F1Z5")
        
        # Add table headers
        p.drawString(100, 500, "Description")
        p.drawString(250, 500, "HSN")
        p.drawString(300, 500, "Qty")
        p.drawString(350, 500, "Price")
        p.drawString(400, 500, "Total")
        
        # Add sample line items
        p.drawString(100, 450, "Product A")
        p.drawString(250, 450, "1234")
        p.drawString(300, 450, "10")
        p.drawString(350, 450, "100.00")
        p.drawString(400, 450, "1000.00")
        
        p.drawString(100, 400, "Product B")
        p.drawString(250, 400, "5678")
        p.drawString(300, 400, "5")
        p.drawString(350, 400, "200.00")
        p.drawString(400, 400, "1000.00")
        
        # Add totals
        p.drawString(300, 300, "Subtotal: 2000.00")
        p.drawString(300, 250, "GST (18%): 360.00")
        p.drawString(300, 200, "Total: 2360.00")
        
        p.showPage()
        p.save()
        
        return buffer.getvalue()
        
    except ImportError:
        # If reportlab is not available, return None
        return None

# Test utilities and helper functions
def assert_confidence_above_threshold(confidence, threshold=None):
    """Assert that confidence score is above threshold."""
    if threshold is None:
        threshold = TestConfig.OCR_CONFIDENCE_THRESHOLD
    assert confidence >= threshold, f"Confidence {confidence} below threshold {threshold}"

def assert_calculation_accurate(expected, actual, tolerance=None):
    """Assert that calculation is accurate within tolerance."""
    if tolerance is None:
        tolerance = TestConfig.CALCULATION_TOLERANCE
    assert abs(expected - actual) <= tolerance, f"Expected {expected}, got {actual}, tolerance {tolerance}"

def create_mock_image():
    """Create a mock image for testing."""
    import numpy as np
    return np.ones((600, 800, 3), dtype=np.uint8) * 255

def create_mock_ocr_result():
    """Create mock OCR result for testing."""
    return TestConfig.SAMPLE_OCR_RESULT.copy()

def create_mock_config():
    """Create mock configuration for testing."""
    from src.utils.config import Config
    config = Config()
    # Override with test-specific settings
    config.ocr_confidence_threshold = TestConfig.OCR_CONFIDENCE_THRESHOLD
    config.calculation_tolerance = TestConfig.CALCULATION_TOLERANCE
    return config