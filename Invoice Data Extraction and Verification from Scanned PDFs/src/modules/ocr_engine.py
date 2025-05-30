"""
Module 3: OCR Engine
Path: src/modules/ocr_engine.py

Purpose: Extract text from preprocessed images using open-source OCR models
Dependencies: pytesseract, easyocr, opencv-python, numpy
"""

import pytesseract
import easyocr
import cv2
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Tuple, Optional, Union
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """
    Handles text extraction from preprocessed images using multiple OCR engines.
    Supports both Tesseract and EasyOCR with confidence scoring.
    """
    
    def __init__(self, primary_engine: str = 'tesseract', fallback_engine: str = 'easyocr',
                 languages: List[str] = ['en'], confidence_threshold: float = 0.6):
        """
        Initialize OCR engine with configuration.
        
        Args:
            primary_engine (str): Primary OCR engine ('tesseract' or 'easyocr')
            fallback_engine (str): Fallback OCR engine
            languages (List[str]): Languages for OCR (default: ['en'])
            confidence_threshold (float): Minimum confidence for text acceptance
        """
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        
        # Initialize engines
        self.tesseract_available = False
        self.easyocr_available = False
        self.easyocr_reader = None
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available OCR engines."""
        # Initialize Tesseract
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.warning(f"Tesseract not available: {str(e)}")
        
        # Initialize EasyOCR
        try:
            self.easyocr_reader = easyocr.Reader(self.languages, gpu=False)
            self.easyocr_available = True
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR not available: {str(e)}")
        
        # Check if any engine is available
        if not self.tesseract_available and not self.easyocr_available:
            raise RuntimeError("No OCR engines available. Install pytesseract or easyocr.")
    
    def extract_text_tesseract(self, image: Image.Image) -> Dict:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image (Image.Image): Preprocessed PIL Image
            
        Returns:
            Dict: OCR results with text, coordinates, and confidence
        """
        if not self.tesseract_available:
            raise RuntimeError("Tesseract not available")
        
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/-:()%'
            
            # Extract text with detailed information
            data = pytesseract.image_to_data(
                cv_image,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            results = {
                'engine': 'tesseract',
                'text_blocks': [],
                'full_text': '',
                'word_count': 0,
                'average_confidence': 0.0
            }
            
            valid_confidences = []
            full_text_parts = []
            
            # Process each detected text element
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i])
                
                if text and confidence > 0:  # Valid text with confidence
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    text_block = {
                        'text': text,
                        'confidence': confidence / 100.0,  # Convert to 0-1 range
                        'bbox': [x, y, x + w, y + h],
                        'level': data['level'][i]
                    }
                    
                    results['text_blocks'].append(text_block)
                    valid_confidences.append(confidence)
                    full_text_parts.append(text)
            
            # Compile full text
            results['full_text'] = ' '.join(full_text_parts)
            results['word_count'] = len([block for block in results['text_blocks'] if len(block['text'].split()) == 1])
            results['average_confidence'] = np.mean(valid_confidences) / 100.0 if valid_confidences else 0.0
            
            logger.info(f"Tesseract extracted {len(results['text_blocks'])} text blocks")
            return results
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            return self._empty_result('tesseract')
    
    def extract_text_easyocr(self, image: Image.Image) -> Dict:
        """
        Extract text using EasyOCR.
        
        Args:
            image (Image.Image): Preprocessed PIL Image
            
        Returns:
            Dict: OCR results with text, coordinates, and confidence
        """
        if not self.easyocr_available:
            raise RuntimeError("EasyOCR not available")
        
        try:
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Perform OCR
            easyocr_results = self.easyocr_reader.readtext(image_array)
            
            # Process results
            results = {
                'engine': 'easyocr',
                'text_blocks': [],
                'full_text': '',
                'word_count': 0,
                'average_confidence': 0.0
            }
            
            confidences = []
            full_text_parts = []
            
            for detection in easyocr_results:
                bbox_points, text, confidence = detection
                
                if confidence >= self.confidence_threshold:
                    # Convert bbox points to standard format [x1, y1, x2, y2]
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    text_block = {
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox,
                        'level': 5  # Word level equivalent
                    }
                    
                    results['text_blocks'].append(text_block)
                    confidences.append(confidence)
                    full_text_parts.append(text.strip())
            
            # Compile results
            results['full_text'] = ' '.join(full_text_parts)
            results['word_count'] = len(full_text_parts)
            results['average_confidence'] = np.mean(confidences) if confidences else 0.0
            
            logger.info(f"EasyOCR extracted {len(results['text_blocks'])} text blocks")
            return results
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {str(e)}")
            return self._empty_result('easyocr')
    
    def _empty_result(self, engine: str) -> Dict:
        """Return empty result structure."""
        return {
            'engine': engine,
            'text_blocks': [],
            'full_text': '',
            'word_count': 0,
            'average_confidence': 0.0
        }
    
    def extract_text_with_fallback(self, image: Image.Image) -> Dict:
        """
        Extract text using primary engine with fallback option.
        
        Args:
            image (Image.Image): Preprocessed PIL Image
            
        Returns:
            Dict: Best OCR results from available engines
        """
        primary_result = None
        fallback_result = None
        
        # Try primary engine
        try:
            if self.primary_engine == 'tesseract' and self.tesseract_available:
                primary_result = self.extract_text_tesseract(image)
            elif self.primary_engine == 'easyocr' and self.easyocr_available:
                primary_result = self.extract_text_easyocr(image)
        except Exception as e:
            logger.warning(f"Primary engine {self.primary_engine} failed: {str(e)}")
        
        # Try fallback engine if primary failed or low confidence
        if (not primary_result or 
            primary_result['average_confidence'] < self.confidence_threshold or
            len(primary_result['text_blocks']) == 0):
            
            try:
                if self.fallback_engine == 'tesseract' and self.tesseract_available:
                    fallback_result = self.extract_text_tesseract(image)
                elif self.fallback_engine == 'easyocr' and self.easyocr_available:
                    fallback_result = self.extract_text_easyocr(image)
            except Exception as e:
                logger.warning(f"Fallback engine {self.fallback_engine} failed: {str(e)}")
        
        # Choose best result
        if primary_result and fallback_result:
            if primary_result['average_confidence'] >= fallback_result['average_confidence']:
                best_result = primary_result
                best_result['fallback_used'] = False
            else:
                best_result = fallback_result
                best_result['fallback_used'] = True
        elif primary_result:
            best_result = primary_result
            best_result['fallback_used'] = False
        elif fallback_result:
            best_result = fallback_result
            best_result['fallback_used'] = True
        else:
            best_result = self._empty_result('none')
            best_result['fallback_used'] = False
        
        logger.info(f"OCR completed using {best_result['engine']}, "
                   f"confidence: {best_result['average_confidence']:.3f}")
        
        return best_result
    
    def extract_text_from_region(self, image: Image.Image, bbox: List[int]) -> Dict:
        """
        Extract text from specific region of image.
        
        Args:
            image (Image.Image): Full image
            bbox (List[int]): Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dict: OCR results from specified region
        """
        try:
            # Crop image to specified region
            x1, y1, x2, y2 = bbox
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Extract text from cropped region
            result = self.extract_text_with_fallback(cropped_image)
            result['region_bbox'] = bbox
            
            return result
            
        except Exception as e:
            logger.error(f"Region OCR failed: {str(e)}")
            return self._empty_result('region')
    
    def process_multiple_images(self, images: List[Image.Image]) -> List[Dict]:
        """
        Process multiple images with OCR.
        
        Args:
            images (List[Image.Image]): List of preprocessed images
            
        Returns:
            List[Dict]: OCR results for each image
        """
        results = []
        
        logger.info(f"Processing {len(images)} images with OCR")
        
        for i, image in enumerate(images):
            try:
                logger.info(f"OCR processing image {i+1}/{len(images)}")
                result = self.extract_text_with_fallback(image)
                result['page_number'] = i + 1
                result['image_size'] = image.size
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {str(e)}")
                error_result = self._empty_result('error')
                error_result['page_number'] = i + 1
                error_result['error'] = str(e)
                results.append(error_result)
        
        logger.info(f"OCR processing completed for {len(results)} images")
        return results
    
    def get_text_by_confidence(self, ocr_result: Dict, min_confidence: float = 0.8) -> List[Dict]:
        """
        Filter text blocks by confidence threshold.
        
        Args:
            ocr_result (Dict): OCR result dictionary
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            List[Dict]: High-confidence text blocks
        """
        if not ocr_result or 'text_blocks' not in ocr_result:
            return []
        
        high_confidence_blocks = [
            block for block in ocr_result['text_blocks']
            if block['confidence'] >= min_confidence
        ]
        
        return high_confidence_blocks
    
    def get_text_in_region(self, ocr_result: Dict, region_bbox: List[int]) -> List[Dict]:
        """
        Get text blocks within specified region.
        
        Args:
            ocr_result (Dict): OCR result dictionary
            region_bbox (List[int]): Region bounding box [x1, y1, x2, y2]
            
        Returns:
            List[Dict]: Text blocks within region
        """
        if not ocr_result or 'text_blocks' not in ocr_result:
            return []
        
        x1, y1, x2, y2 = region_bbox
        region_blocks = []
        
        for block in ocr_result['text_blocks']:
            bx1, by1, bx2, by2 = block['bbox']
            
            # Check if block overlaps with region
            if (bx1 < x2 and bx2 > x1 and by1 < y2 and by2 > y1):
                region_blocks.append(block)
        
        return region_blocks
    
    def combine_ocr_results(self, results_list: List[Dict]) -> Dict:
        """
        Combine OCR results from multiple pages.
        
        Args:
            results_list (List[Dict]): List of OCR results
            
        Returns:
            Dict: Combined OCR results
        """
        combined = {
            'pages': results_list,
            'total_pages': len(results_list),
            'combined_text': '',
            'total_word_count': 0,
            'average_confidence': 0.0,
            'engines_used': set()
        }
        
        all_text_parts = []
        all_confidences = []
        total_words = 0
        
        for result in results_list:
            if result.get('full_text'):
                all_text_parts.append(result['full_text'])
            
            if result.get('average_confidence', 0) > 0:
                all_confidences.append(result['average_confidence'])
            
            total_words += result.get('word_count', 0)
            
            if 'engine' in result:
                combined['engines_used'].add(result['engine'])
        
        # Compile combined results
        combined['combined_text'] = '\n\n'.join(all_text_parts)
        combined['total_word_count'] = total_words
        combined['average_confidence'] = np.mean(all_confidences) if all_confidences else 0.0
        combined['engines_used'] = list(combined['engines_used'])
        
        return combined
    
    def save_ocr_results(self, results: Union[Dict, List[Dict]], output_path: str) -> bool:
        """
        Save OCR results to JSON file.
        
        Args:
            results (Union[Dict, List[Dict]]): OCR results to save
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            import json
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"OCR results saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save OCR results: {str(e)}")
            return False


def main():
    """
    Example usage and testing of OCREngine.
    """
    # Initialize OCR engine
    ocr_engine = OCREngine(
        primary_engine='tesseract',
        fallback_engine='easyocr',
        languages=['en'],
        confidence_threshold=0.6
    )
    
    try:
        # Test with sample processed image
        sample_image = Image.open("processed_sample.png")
        
        # Extract text
        result = ocr_engine.extract_text_with_fallback(sample_image)
        
        print(f"OCR Engine: {result['engine']}")
        print(f"Average Confidence: {result['average_confidence']:.3f}")
        print(f"Word Count: {result['word_count']}")
        print(f"Text Blocks: {len(result['text_blocks'])}")
        print(f"\nExtracted Text:\n{result['full_text']}")
        
        # Get high confidence text
        high_conf_blocks = ocr_engine.get_text_by_confidence(result, min_confidence=0.8)
        print(f"\nHigh confidence blocks: {len(high_conf_blocks)}")
        
        # Save results
        ocr_engine.save_ocr_results(result, "ocr_results.json")
        
    except FileNotFoundError:
        print("Sample image not found. This module requires input from Module 2.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()