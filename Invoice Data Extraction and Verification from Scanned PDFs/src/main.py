#!/usr/bin/env python3
"""
Invoice Data Extraction & Verification System
Main orchestrator script for processing scanned invoice PDFs
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Fix import paths - add current directory and parent directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

try:
    # Import modules using relative imports (since we're running from src/)
    from modules.pdf_processor import PDFProcessor
    from modules.image_preprocessor import ImagePreprocessor
    from modules.ocr_engine import OCREngine
    from modules.field_extractor import FieldExtractor
    from modules.table_parser import TableParser
    from modules.seal_detector import SealDetector
    from modules.verifier import DataVerifier as Verifier
    from modules.output_generator import OutputGenerator
    # Import utilities
    from utils.config import (
        INPUT_DIR, OUTPUT_DIR, SEAL_DIR, 
        DEFAULT_OCR_ENGINE, CONFIDENCE_THRESHOLD
    )
    from utils.logger import setup_logger
    from utils.helpers import ensure_directories, get_pdf_files
    from src.utils.logger import get_logger, setup_logger, InvoiceLogger

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Python path: {sys.path}")
    print("\n🔧 To fix this, make sure you:")
    print("1. Run from project root: python src/main.py")
    print("2. OR run from src directory: python main.py (with fixed imports)")
    print("3. Make sure all module files exist in src/modules/")
    sys.exit(1)


class InvoiceExtractionPipeline:
    """Main pipeline class for invoice data extraction and verification"""
    
    def __init__(self):
        """Initialize the extraction pipeline"""
        # Setup logging
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.image_preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(engine_type=DEFAULT_OCR_ENGINE)
        self.field_extractor = FieldExtractor()
        self.table_parser = TableParser()
        self.seal_detector = SealDetector()
        self.verifier = Verifier()
        self.output_generator = OutputGenerator()
        
        # Ensure output directories exist
        ensure_directories()
        
        self.logger.info("Invoice extraction pipeline initialized successfully")
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file through the complete extraction pipeline
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted data and verification results
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Step 1: Convert PDF to images
            self.logger.info("Step 1: Converting PDF to images...")
            images = self.pdf_processor.extract_images_from_pdf(pdf_path)
            
            # Step 2: Preprocess images
            self.logger.info("Step 2: Preprocessing images...")
            preprocessed_images = []
            for img in images:
                processed_img = self.image_preprocessor.enhance_image(img)
                preprocessed_images.append(processed_img)
            
            # Step 3: Extract text using OCR
            self.logger.info("Step 3: Extracting text with OCR...")
            ocr_results = []
            for img in preprocessed_images:
                ocr_data = self.ocr_engine.extract_text_with_coordinates(img)
                ocr_results.append(ocr_data)
            
            # Step 4: Extract structured fields
            self.logger.info("Step 4: Extracting structured fields...")
            extracted_fields = self.field_extractor.extract_all_fields(ocr_results)
            
            # Step 5: Parse table data
            self.logger.info("Step 5: Parsing table data...")
            table_data = self.table_parser.extract_line_items(
                preprocessed_images, ocr_results
            )
            
            # Step 6: Detect seals and signatures
            self.logger.info("Step 6: Detecting seals and signatures...")
            seal_results = self.seal_detector.detect_and_extract_seals(
                preprocessed_images, pdf_path
            )
            
            # Step 7: Combine all extracted data
            combined_data = {
                **extracted_fields,
                'line_items': table_data,
                'seal_and_sign_present': seal_results['detected'],
                'seal_images': seal_results['image_paths']
            }
            
            # Step 8: Verify and validate data
            self.logger.info("Step 7: Verifying and validating data...")
            verification_results = self.verifier.verify_invoice_data(combined_data)
            
            # Step 9: Generate output files
            self.logger.info("Step 8: Generating output files...")
            output_paths = self.output_generator.generate_all_outputs(
                combined_data, verification_results, pdf_path
            )
            
            self.logger.info(f"✅ Successfully processed: {pdf_path}")
            
            return {
                'extracted_data': combined_data,
                'verification_results': verification_results,
                'output_paths': output_paths,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {pdf_path}: {str(e)}")
            return {
                'extracted_data': None,
                'verification_results': None,
                'output_paths': None,
                'status': 'error',
                'error_message': str(e)
            }
    
    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Process all PDF files in the input directory
        
        Returns:
            List of results for each processed PDF
        """
        pdf_files = get_pdf_files(INPUT_DIR)
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {INPUT_DIR}")
            print(f"⚠️  No PDF files found in {INPUT_DIR}")
            return []
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        print(f"📄 Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            print(f"\n🔄 Processing: {os.path.basename(pdf_file)}")
            result = self.process_single_pdf(pdf_file)
            results.append({
                'file': pdf_file,
                'result': result
            })
            
            if result['status'] == 'success':
                print(f"✅ Successfully processed: {os.path.basename(pdf_file)}")
            else:
                print(f"❌ Failed to process: {os.path.basename(pdf_file)}")
                print(f"   Error: {result.get('error_message', 'Unknown error')}")
        
        return results


def main():
    """Main function to run the invoice extraction pipeline"""
    print("🚀 Starting Invoice Data Extraction & Verification System")
    print("=" * 60)
    
    try:
        # Initialize the pipeline
        pipeline = InvoiceExtractionPipeline()
        
        # Process all PDFs
        results = pipeline.process_all_pdfs()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        
        if not results:
            print("⚠️  No files were processed")
            return
        
        successful = sum(1 for r in results if r['result']['status'] == 'success')
        failed = len(results) - successful
        
        print(f"📄 Total files processed: {len(results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if successful > 0:
            print(f"\n📁 Output files generated in: {OUTPUT_DIR}")
            print("   - extracted_data.json")
            print("   - extracted_data.xlsx") 
            print("   - verifiability_report.json")
            print(f"   - seal_signatures/ (if any seals detected)")
        
        print("\n🎉 Processing completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logging.exception("Fatal error in main pipeline")


if __name__ == "__main__":
    main()