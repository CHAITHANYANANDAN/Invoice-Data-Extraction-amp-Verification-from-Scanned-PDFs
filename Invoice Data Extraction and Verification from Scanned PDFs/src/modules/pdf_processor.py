"""
Module 1: PDF Processor
Path: src/modules/pdf_processor.py

Purpose: Convert scanned PDF pages to images for OCR processing
Dependencies: pdf2image, Pillow
"""

import os
import tempfile
from typing import List, Tuple, Optional
from PIL import Image
import logging

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
except ImportError:
    raise ImportError("pdf2image is required. Install with: pip install pdf2image")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles conversion of PDF files to images for OCR processing.
    Supports multiple pages and handles various PDF formats.
    """
    
    def __init__(self, dpi: int = 300, output_format: str = 'PNG'):
        """
        Initialize PDF processor with configuration.
        
        Args:
            dpi (int): Resolution for image conversion (default: 300)
            output_format (str): Output image format (default: 'PNG')
        """
        self.dpi = dpi
        self.output_format = output_format.upper()
        self.temp_dir = None
        
        # Validate output format
        if self.output_format not in ['PNG', 'JPEG', 'TIFF']:
            raise ValueError("Supported formats: PNG, JPEG, TIFF")
    
    def extract_images_from_pdf(self, pdf_path: str) -> Tuple[List[Image.Image], bool]:
        """
        Convert PDF pages to PIL Image objects.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Tuple[List[Image.Image], bool]: List of PIL Images and success status
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: For PDF processing errors
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not os.path.isfile(pdf_path):
            raise ValueError(f"Path is not a file: {pdf_path}")
        
        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            logger.warning(f"File may not be a PDF: {pdf_path}")
        
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            logger.info(f"Using DPI: {self.dpi}")
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                output_folder=None,  # Keep in memory
                first_page=None,
                last_page=None,
                fmt=self.output_format.lower(),
                thread_count=1,
                userpw=None,
                use_cropbox=False,
                strict=False
            )
            
            if not images:
                logger.error("No pages found in PDF")
                return [], False
            
            logger.info(f"Successfully converted {len(images)} pages")
            
            # Log image details
            for i, img in enumerate(images):
                logger.info(f"Page {i+1}: Size={img.size}, Mode={img.mode}")
            
            return images, True
            
        except PDFInfoNotInstalledError:
            error_msg = "poppler-utils not installed. Please install poppler-utils for PDF processing."
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except PDFPageCountError as e:
            error_msg = f"Error reading PDF page count: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except PDFSyntaxError as e:
            error_msg = f"PDF syntax error: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error processing PDF: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def save_page_images(self, images: List[Image.Image], output_dir: str, 
                        filename_prefix: str = "page") -> List[str]:
        """
        Save PIL Images to disk with organized naming.
        
        Args:
            images (List[Image.Image]): List of PIL Images
            output_dir (str): Directory to save images
            filename_prefix (str): Prefix for saved filenames
            
        Returns:
            List[str]: List of saved image file paths
            
        Raises:
            Exception: For file I/O errors
        """
        if not images:
            logger.warning("No images to save")
            return []
        
        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to create output directory {output_dir}: {str(e)}")
        
        saved_paths = []
        
        try:
            for i, image in enumerate(images):
                # Generate filename
                page_num = str(i + 1).zfill(3)  # Zero-padded page numbers
                filename = f"{filename_prefix}_{page_num}.{self.output_format.lower()}"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                image.save(filepath, format=self.output_format, optimize=True)
                saved_paths.append(filepath)
                
                logger.info(f"Saved page {i+1}: {filepath}")
            
            logger.info(f"Successfully saved {len(saved_paths)} images to {output_dir}")
            return saved_paths
            
        except Exception as e:
            error_msg = f"Error saving images: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def create_temp_directory(self) -> str:
        """
        Create a temporary directory for intermediate processing.
        
        Returns:
            str: Path to temporary directory
        """
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="pdf_processor_")
            logger.info(f"Created temporary directory: {self.temp_dir}")
            return self.temp_dir
        except Exception as e:
            raise Exception(f"Failed to create temporary directory: {str(e)}")
    
    def cleanup_temp_directory(self) -> None:
        """
        Clean up temporary directory and files.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
                self.temp_dir = None
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory: {str(e)}")
    
    def process_pdf_to_images(self, pdf_path: str, output_dir: Optional[str] = None,
                             save_images: bool = False) -> Tuple[List[Image.Image], List[str]]:
        """
        Complete PDF to images processing pipeline.
        
        Args:
            pdf_path (str): Path to PDF file
            output_dir (Optional[str]): Directory to save images (if save_images=True)
            save_images (bool): Whether to save images to disk
            
        Returns:
            Tuple[List[Image.Image], List[str]]: PIL Images and saved file paths
        """
        try:
            # Extract images from PDF
            images, success = self.extract_images_from_pdf(pdf_path)
            
            if not success or not images:
                return [], []
            
            saved_paths = []
            
            # Save images if requested
            if save_images and output_dir:
                # Generate filename prefix from PDF name
                pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
                filename_prefix = f"{pdf_basename}_page"
                
                saved_paths = self.save_page_images(images, output_dir, filename_prefix)
            
            return images, saved_paths
            
        except Exception as e:
            logger.error(f"PDF processing pipeline failed: {str(e)}")
            raise
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get basic information about the PDF file.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            dict: PDF information including page count, file size
        """
        info = {
            'file_path': pdf_path,
            'file_exists': False,
            'file_size_mb': 0,
            'page_count': 0,
            'error': None
        }
        
        try:
            if os.path.exists(pdf_path):
                info['file_exists'] = True
                info['file_size_mb'] = round(os.path.getsize(pdf_path) / (1024 * 1024), 2)
                
                # Get page count by attempting conversion
                images, success = self.extract_images_from_pdf(pdf_path)
                if success:
                    info['page_count'] = len(images)
                else:
                    info['error'] = "Failed to read PDF pages"
            else:
                info['error'] = "File not found"
                
        except Exception as e:
            info['error'] = str(e)
        
        return info


def main():
    """
    Example usage and testing of PDFProcessor.
    """
    # Example usage
    processor = PDFProcessor(dpi=300, output_format='PNG')
    
    # Test with sample PDF (replace with actual path)
    pdf_path = "input/sample_invoice.pdf"
    output_dir = "temp_images"
    
    try:
        # Get PDF info
        info = processor.get_pdf_info(pdf_path)
        print(f"PDF Info: {info}")
        
        if info['file_exists'] and info['page_count'] > 0:
            # Process PDF to images
            images, saved_paths = processor.process_pdf_to_images(
                pdf_path=pdf_path,
                output_dir=output_dir,
                save_images=True
            )
            
            print(f"Processed {len(images)} pages")
            print(f"Saved files: {saved_paths}")
            
            # Example: Access first page image
            if images:
                first_page = images[0]
                print(f"First page - Size: {first_page.size}, Mode: {first_page.mode}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        processor.cleanup_temp_directory()


if __name__ == "__main__":
    main()