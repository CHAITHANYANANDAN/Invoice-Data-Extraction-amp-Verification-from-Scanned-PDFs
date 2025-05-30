"""
Module 2: Image Preprocessor
Path: src/modules/image_preprocessor.py

Purpose: Enhance image quality for better OCR results through various preprocessing techniques
Dependencies: opencv-python, numpy, Pillow, scikit-image
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import List, Tuple, Optional, Union
from skimage import filters, morphology, restoration
from skimage.measure import label, regionprops
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing and enhancement for optimal OCR performance.
    Implements various techniques including denoising, binarization, contrast enhancement, and deskewing.
    """
    
    def __init__(self, target_dpi: int = 300, enhance_contrast: bool = True, 
                 remove_noise: bool = True, auto_deskew: bool = True):
        """
        Initialize image preprocessor with configuration.
        
        Args:
            target_dpi (int): Target DPI for processing (default: 300)
            enhance_contrast (bool): Whether to enhance contrast (default: True)
            remove_noise (bool): Whether to apply noise removal (default: True)
            auto_deskew (bool): Whether to automatically deskew images (default: True)
        """
        self.target_dpi = target_dpi
        self.enhance_contrast = enhance_contrast
        self.remove_noise = remove_noise
        self.auto_deskew = auto_deskew
        
        # Processing parameters
        self.noise_kernel_size = 3
        self.morph_kernel_size = 2
        self.gaussian_blur_sigma = 0.5
        self.contrast_factor = 1.2
        self.brightness_factor = 1.1
        
    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format.
        
        Args:
            pil_image (Image.Image): PIL Image object
            
        Returns:
            np.ndarray: OpenCV image array
        """
        # Convert PIL to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        cv2_image = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        
        return cv2_image
    
    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV format to PIL Image.
        
        Args:
            cv2_image (np.ndarray): OpenCV image array
            
        Returns:
            Image.Image: PIL Image object
        """
        # Handle different image formats
        if len(cv2_image.shape) == 3:
            # Color image - convert BGR to RGB
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale image
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    def denoise_image(self, image: Image.Image, method: str = 'bilateral') -> Image.Image:
        """
        Remove noise from image using various denoising techniques.
        
        Args:
            image (Image.Image): Input PIL Image
            method (str): Denoising method ('bilateral', 'gaussian', 'median', 'nlm')
            
        Returns:
            Image.Image: Denoised PIL Image
        """
        try:
            # Convert to OpenCV format
            cv2_img = self.pil_to_cv2(image)
            
            if method == 'bilateral':
                # Bilateral filter - good for preserving edges while removing noise
                denoised = cv2.bilateralFilter(cv2_img, 9, 75, 75)
                
            elif method == 'gaussian':
                # Gaussian blur - simple noise reduction
                denoised = cv2.GaussianBlur(cv2_img, (5, 5), self.gaussian_blur_sigma)
                
            elif method == 'median':
                # Median filter - good for salt and pepper noise
                denoised = cv2.medianBlur(cv2_img, 5)
                
            elif method == 'nlm':
                # Non-local means denoising - advanced technique
                denoised = cv2.fastNlMeansDenoisingColored(cv2_img, None, 10, 10, 7, 21)
                
            else:
                logger.warning(f"Unknown denoising method: {method}. Using bilateral.")
                denoised = cv2.bilateralFilter(cv2_img, 9, 75, 75)
            
            # Convert back to PIL
            result = self.cv2_to_pil(denoised)
            logger.info(f"Applied {method} denoising")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in denoising: {str(e)}")
            return image
    
    def enhance_contrast_and_brightness(self, image: Image.Image) -> Image.Image:
        """
        Enhance image contrast and brightness for better text visibility.
        
        Args:
            image (Image.Image): Input PIL Image
            
        Returns:
            Image.Image: Enhanced PIL Image
        """
        try:
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            enhanced = contrast_enhancer.enhance(self.contrast_factor)
            
            # Enhance brightness
            brightness_enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = brightness_enhancer.enhance(self.brightness_factor)
            
            # Enhance sharpness slightly
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.1)
            
            logger.info("Enhanced contrast and brightness")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in contrast enhancement: {str(e)}")
            return image
    
    def binarize_image(self, image: Image.Image, method: str = 'otsu') -> Image.Image:
        """
        Convert image to binary (black and white) for optimal OCR.
        
        Args:
            image (Image.Image): Input PIL Image
            method (str): Binarization method ('otsu', 'adaptive', 'threshold')
            
        Returns:
            Image.Image: Binarized PIL Image
        """
        try:
            # Convert to grayscale first
            gray_image = image.convert('L')
            cv2_gray = np.array(gray_image)
            
            if method == 'otsu':
                # Otsu's thresholding - automatically finds optimal threshold
                _, binary = cv2.threshold(cv2_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif method == 'adaptive':
                # Adaptive thresholding - good for varying lighting conditions
                binary = cv2.adaptiveThreshold(
                    cv2_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
            elif method == 'threshold':
                # Simple thresholding with fixed value
                _, binary = cv2.threshold(cv2_gray, 127, 255, cv2.THRESH_BINARY)
                
            else:
                logger.warning(f"Unknown binarization method: {method}. Using Otsu.")
                _, binary = cv2.threshold(cv2_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            result = Image.fromarray(binary)
            logger.info(f"Applied {method} binarization")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in binarization: {str(e)}")
            return image
    
    def deskew_image(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """
        Detect and correct skew in scanned documents.
        
        Args:
            image (Image.Image): Input PIL Image
            
        Returns:
            Tuple[Image.Image, float]: Deskewed image and detected angle
        """
        try:
            # Convert to OpenCV grayscale
            cv2_img = self.pil_to_cv2(image)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough Line Transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate angles of detected lines
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    if -45 <= angle <= 45:  # Focus on reasonable skew angles
                        angles.append(angle)
                
                if angles:
                    # Use median angle to avoid outliers
                    skew_angle = np.median(angles)
                    
                    # Only deskew if angle is significant (> 0.5 degrees)
                    if abs(skew_angle) > 0.5:
                        # Rotate image to correct skew
                        rotated = image.rotate(-skew_angle, expand=True, fillcolor='white')
                        logger.info(f"Deskewed image by {skew_angle:.2f} degrees")
                        return rotated, skew_angle
                    else:
                        logger.info("No significant skew detected")
                        return image, 0.0
                else:
                    logger.info("No valid skew angles detected")
                    return image, 0.0
            else:
                logger.info("No lines detected for skew correction")
                return image, 0.0
                
        except Exception as e:
            logger.error(f"Error in deskewing: {str(e)}")
            return image, 0.0
    
    def remove_borders_and_shadows(self, image: Image.Image) -> Image.Image:
        """
        Remove borders and shadows that may interfere with OCR.
        
        Args:
            image (Image.Image): Input PIL Image
            
        Returns:
            Image.Image: Processed PIL Image
        """
        try:
            # Convert to OpenCV
            cv2_img = self.pil_to_cv2(image)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological operations to remove small artifacts
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Convert back to color
            cleaned_color = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            
            # Convert back to PIL
            result = self.cv2_to_pil(cleaned_color)
            logger.info("Removed borders and shadows")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in border removal: {str(e)}")
            return image
    
    def resize_for_ocr(self, image: Image.Image, min_height: int = 1200) -> Image.Image:
        """
        Resize image to optimal size for OCR processing.
        
        Args:
            image (Image.Image): Input PIL Image
            min_height (int): Minimum height for OCR (default: 1200)
            
        Returns:
            Image.Image: Resized PIL Image
        """
        try:
            width, height = image.size
            
            # Only resize if image is too small
            if height < min_height:
                scale_factor = min_height / height
                new_width = int(width * scale_factor)
                new_height = min_height
                
                # Use high-quality resampling
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                
                return resized
            else:
                logger.info("Image size is already optimal for OCR")
                return image
                
        except Exception as e:
            logger.error(f"Error in resizing: {str(e)}")
            return image
    
    def preprocess_image(self, image: Image.Image, 
                        custom_pipeline: Optional[List[str]] = None) -> Tuple[Image.Image, dict]:
        """
        Complete image preprocessing pipeline.
        
        Args:
            image (Image.Image): Input PIL Image
            custom_pipeline (Optional[List[str]]): Custom processing steps
            
        Returns:
            Tuple[Image.Image, dict]: Processed image and processing metadata
        """
        try:
            processed_image = image.copy()
            metadata = {
                'original_size': image.size,
                'original_mode': image.mode,
                'processing_steps': [],
                'skew_angle': 0.0,
                'processing_successful': True
            }
            
            # Define default pipeline
            if custom_pipeline is None:
                pipeline = [
                    'resize',
                    'denoise',
                    'enhance_contrast',
                    'deskew',
                    'remove_borders',
                    'binarize'
                ]
            else:
                pipeline = custom_pipeline
            
            logger.info(f"Starting preprocessing pipeline: {pipeline}")
            
            # Execute pipeline steps
            for step in pipeline:
                try:
                    if step == 'resize':
                        processed_image = self.resize_for_ocr(processed_image)
                        metadata['processing_steps'].append('resize')
                        
                    elif step == 'denoise' and self.remove_noise:
                        processed_image = self.denoise_image(processed_image, method='bilateral')
                        metadata['processing_steps'].append('denoise')
                        
                    elif step == 'enhance_contrast' and self.enhance_contrast:
                        processed_image = self.enhance_contrast_and_brightness(processed_image)
                        metadata['processing_steps'].append('enhance_contrast')
                        
                    elif step == 'deskew' and self.auto_deskew:
                        processed_image, skew_angle = self.deskew_image(processed_image)
                        metadata['skew_angle'] = skew_angle
                        metadata['processing_steps'].append('deskew')
                        
                    elif step == 'remove_borders':
                        processed_image = self.remove_borders_and_shadows(processed_image)
                        metadata['processing_steps'].append('remove_borders')
                        
                    elif step == 'binarize':
                        processed_image = self.binarize_image(processed_image, method='otsu')
                        metadata['processing_steps'].append('binarize')
                        
                    else:
                        logger.warning(f"Unknown or skipped processing step: {step}")
                        
                except Exception as e:
                    logger.error(f"Error in processing step '{step}': {str(e)}")
                    metadata['processing_successful'] = False
            
            # Update final metadata
            metadata['final_size'] = processed_image.size
            metadata['final_mode'] = processed_image.mode
            
            logger.info(f"Preprocessing completed. Steps applied: {metadata['processing_steps']}")
            
            return processed_image, metadata
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            metadata['processing_successful'] = False
            return image, metadata
    
    def process_multiple_images(self, images: List[Image.Image]) -> Tuple[List[Image.Image], List[dict]]:
        """
        Process multiple images with the same preprocessing pipeline.
        
        Args:
            images (List[Image.Image]): List of PIL Images
            
        Returns:
            Tuple[List[Image.Image], List[dict]]: Processed images and metadata
        """
        processed_images = []
        all_metadata = []
        
        logger.info(f"Processing {len(images)} images")
        
        for i, image in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{len(images)}")
                processed_img, metadata = self.preprocess_image(image)
                
                processed_images.append(processed_img)
                all_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {str(e)}")
                # Add original image and error metadata
                processed_images.append(image)
                all_metadata.append({
                    'processing_successful': False,
                    'error': str(e),
                    'original_size': image.size,
                    'original_mode': image.mode
                })
        
        logger.info(f"Completed processing {len(processed_images)} images")
        return processed_images, all_metadata
    
    def save_processed_images(self, images: List[Image.Image], output_dir: str, 
                             filename_prefix: str = "processed") -> List[str]:
        """
        Save processed images to disk.
        
        Args:
            images (List[Image.Image]): List of processed PIL Images
            output_dir (str): Output directory
            filename_prefix (str): Filename prefix
            
        Returns:
            List[str]: List of saved file paths
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        for i, image in enumerate(images):
            try:
                filename = f"{filename_prefix}_page_{str(i+1).zfill(3)}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save with high quality
                image.save(filepath, 'PNG', optimize=True)
                saved_paths.append(filepath)
                
                logger.info(f"Saved processed image: {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving image {i+1}: {str(e)}")
        
        return saved_paths


def main():
    """
    Example usage and testing of ImagePreprocessor.
    """
    # Example usage
    preprocessor = ImagePreprocessor(
        target_dpi=300,
        enhance_contrast=True,
        remove_noise=True,
        auto_deskew=True
    )
    
    # Test with sample image (this would normally come from Module 1)
    try:
        # Load sample image
        sample_image = Image.open("temp_images/sample_page_001.png")
        
        # Process single image
        processed_image, metadata = preprocessor.preprocess_image(sample_image)
        
        print(f"Processing metadata: {metadata}")
        print(f"Original size: {metadata['original_size']}")
        print(f"Final size: {metadata['final_size']}")
        print(f"Processing steps: {metadata['processing_steps']}")
        print(f"Skew angle: {metadata['skew_angle']}")
        
        # Save processed image
        processed_image.save("processed_sample.png")
        print("Saved processed image: processed_sample.png")
        
    except FileNotFoundError:
        print("Sample image not found. This module requires input from Module 1.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()