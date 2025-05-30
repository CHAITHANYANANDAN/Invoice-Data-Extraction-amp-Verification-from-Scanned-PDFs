"""
Module 6: Seal Detector
Purpose: Detect and extract seals/signatures from invoices using image processing techniques
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any
from PIL import Image
import logging

class SealDetector:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the seal detector with configuration parameters
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {}
        self.min_contour_area = self.config.get('min_contour_area', 500)
        self.max_contour_area = self.config.get('max_contour_area', 50000)
        self.aspect_ratio_threshold = self.config.get('aspect_ratio_threshold', 0.3)
        self.circularity_threshold = self.config.get('circularity_threshold', 0.4)
        self.logger = logging.getLogger(__name__)
        
    def detect_seals_signatures(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect seal and signature regions in the invoice image
        
        Args:
            image: Input invoice image as numpy array
            
        Returns:
            List of detected seal/signature regions with metadata
        """
        try:
            self.logger.info("Starting seal/signature detection")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            detected_regions = []
            
            # Method 1: Detect circular seals
            circular_seals = self._detect_circular_seals(gray)
            detected_regions.extend(circular_seals)
            
            # Method 2: Detect rectangular signature boxes
            signature_boxes = self._detect_signature_boxes(gray)
            detected_regions.extend(signature_boxes)
            
            # Method 3: Detect hand-drawn signatures
            signatures = self._detect_handwritten_signatures(gray)
            detected_regions.extend(signatures)
            
            # Remove overlapping detections
            filtered_regions = self._remove_overlapping_regions(detected_regions)
            
            self.logger.info(f"Detected {len(filtered_regions)} seal/signature regions")
            return filtered_regions
            
        except Exception as e:
            self.logger.error(f"Error in seal detection: {str(e)}")
            return []
    
    def _detect_circular_seals(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect circular/elliptical seals using contour analysis
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            List of detected circular seal regions
        """
        circular_seals = []
        
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if self.min_contour_area < area < self.max_contour_area:
                    # Calculate circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > self.circularity_threshold:
                            # Get bounding rectangle
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            circular_seals.append({
                                'type': 'circular_seal',
                                'bbox': [x, y, x + w, y + h],
                                'area': area,
                                'circularity': circularity,
                                'confidence': min(circularity * 2, 1.0),
                                'contour': contour
                            })
            
            self.logger.info(f"Detected {len(circular_seals)} circular seals")
            
        except Exception as e:
            self.logger.error(f"Error detecting circular seals: {str(e)}")
        
        return circular_seals
    
    def _detect_signature_boxes(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect rectangular signature boxes or stamps
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            List of detected signature box regions
        """
        signature_boxes = []
        
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_contour_area < area < self.max_contour_area:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular (4-8 vertices)
                    if 4 <= len(approx) <= 8:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h
                        
                        # Check aspect ratio (not too thin/tall)
                        if 0.2 < aspect_ratio < 5.0:
                            # Check if region contains significant content
                            roi = gray_image[y:y+h, x:x+w]
                            content_density = self._calculate_content_density(roi)
                            
                            if content_density > 0.1:  # At least 10% non-white pixels
                                signature_boxes.append({
                                    'type': 'signature_box',
                                    'bbox': [x, y, x + w, y + h],
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'vertices': len(approx),
                                    'content_density': content_density,
                                    'confidence': min(content_density * 2, 1.0),
                                    'contour': contour
                                })
            
            self.logger.info(f"Detected {len(signature_boxes)} signature boxes")
            
        except Exception as e:
            self.logger.error(f"Error detecting signature boxes: {str(e)}")
        
        return signature_boxes
    
    def _detect_handwritten_signatures(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect handwritten signatures using stroke analysis
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            List of detected handwritten signature regions
        """
        signatures = []
        
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
            
            # Adaptive threshold for handwriting
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY_INV, 15, 5
            )
            
            # Morphological operations to connect strokes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if 200 < area < 20000:  # Signature-sized areas
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Signatures tend to be wider than tall
                    if 1.5 < aspect_ratio < 8.0:
                        # Calculate stroke characteristics
                        roi = thresh[y:y+h, x:x+w]
                        stroke_density = np.sum(roi > 0) / (w * h)
                        
                        # Check for signature-like characteristics
                        if 0.05 < stroke_density < 0.4:  # Not too dense, not too sparse
                            # Calculate contour complexity (signatures have curves)
                            hull = cv2.convexHull(contour)
                            hull_area = cv2.contourArea(hull)
                            solidity = area / hull_area if hull_area > 0 else 0
                            
                            if solidity < 0.8:  # Signatures are not solid shapes
                                signatures.append({
                                    'type': 'handwritten_signature',
                                    'bbox': [x, y, x + w, y + h],
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'stroke_density': stroke_density,
                                    'solidity': solidity,
                                    'confidence': min((1 - solidity) * stroke_density * 3, 1.0),
                                    'contour': contour
                                })
            
            self.logger.info(f"Detected {len(signatures)} handwritten signatures")
            
        except Exception as e:
            self.logger.error(f"Error detecting handwritten signatures: {str(e)}")
        
        return signatures
    
    def _calculate_content_density(self, roi: np.ndarray) -> float:
        """
        Calculate the density of content (non-white pixels) in a region
        
        Args:
            roi: Region of interest
            
        Returns:
            Content density ratio (0-1)
        """
        if roi.size == 0:
            return 0.0
        
        # Threshold to separate content from background
        _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
        content_pixels = np.sum(binary > 0)
        total_pixels = roi.size
        
        return content_pixels / total_pixels
    
    def _remove_overlapping_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove overlapping detected regions, keeping the one with higher confidence
        
        Args:
            regions: List of detected regions
            
        Returns:
            Filtered list of non-overlapping regions
        """
        if len(regions) <= 1:
            return regions
        
        # Sort by confidence (descending)
        sorted_regions = sorted(regions, key=lambda x: x['confidence'], reverse=True)
        filtered_regions = []
        
        for current_region in sorted_regions:
            is_overlapping = False
            current_bbox = current_region['bbox']
            
            for existing_region in filtered_regions:
                existing_bbox = existing_region['bbox']
                
                # Calculate overlap
                overlap_ratio = self._calculate_overlap_ratio(current_bbox, existing_bbox)
                
                if overlap_ratio > 0.3:  # 30% overlap threshold
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_regions.append(current_region)
        
        return filtered_regions
    
    def _calculate_overlap_ratio(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate the overlap ratio between two bounding boxes
        
        Args:
            bbox1, bbox2: Bounding boxes as [x1, y1, x2, y2]
            
        Returns:
            Overlap ratio (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def crop_and_save_seals(self, image: np.ndarray, regions: List[Dict[str, Any]], 
                           output_dir: str) -> List[str]:
        """
        Crop detected seal/signature regions and save them as separate images
        
        Args:
            image: Original invoice image
            regions: List of detected regions
            output_dir: Directory to save cropped images
            
        Returns:
            List of saved image file paths
        """
        saved_files = []
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            for i, region in enumerate(regions):
                bbox = region['bbox']
                x1, y1, x2, y2 = bbox
                
                # Crop the region with some padding
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                cropped_region = image[y1:y2, x1:x2]
                
                if cropped_region.size > 0:
                    # Generate filename
                    region_type = region['type']
                    confidence = region['confidence']
                    filename = f"{region_type}_{i+1}_conf_{confidence:.2f}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save the cropped image
                    if len(cropped_region.shape) == 3:
                        # Convert BGR to RGB for PIL
                        cropped_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
                    
                    pil_image = Image.fromarray(cropped_region)
                    pil_image.save(filepath)
                    
                    saved_files.append(filepath)
                    self.logger.info(f"Saved {region_type} to {filepath}")
            
            self.logger.info(f"Saved {len(saved_files)} seal/signature images")
            
        except Exception as e:
            self.logger.error(f"Error saving seal images: {str(e)}")
        
        return saved_files
    
    def validate_seal_presence(self, detection_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the presence and quality of detected seals/signatures
        
        Args:
            detection_results: List of detected regions
            
        Returns:
            Validation results with overall assessment
        """
        validation_result = {
            'has_seals': len(detection_results) > 0,
            'total_detections': len(detection_results),
            'seal_types': {},
            'highest_confidence': 0.0,
            'average_confidence': 0.0,
            'validation_status': 'no_seals_detected'
        }
        
        if not detection_results:
            return validation_result
        
        # Analyze detection types and confidence
        type_counts = {}
        confidences = []
        
        for region in detection_results:
            region_type = region['type']
            confidence = region['confidence']
            
            type_counts[region_type] = type_counts.get(region_type, 0) + 1
            confidences.append(confidence)
        
        validation_result['seal_types'] = type_counts
        validation_result['highest_confidence'] = max(confidences)
        validation_result['average_confidence'] = sum(confidences) / len(confidences)
        
        # Determine validation status
        if validation_result['highest_confidence'] > 0.7:
            validation_result['validation_status'] = 'high_confidence_seals'
        elif validation_result['highest_confidence'] > 0.4:
            validation_result['validation_status'] = 'medium_confidence_seals'
        else:
            validation_result['validation_status'] = 'low_confidence_seals'
        
        return validation_result


def main():
    """
    Test function for the seal detector module
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python seal_detector.py <image_path>")
        return
    
    # Test the seal detector
    detector = SealDetector()
    
    # Load test image
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Detect seals and signatures
    regions = detector.detect_seals_signatures(image)
    
    # Save cropped regions
    output_dir = "test_seal_output"
    saved_files = detector.crop_and_save_seals(image, regions, output_dir)
    
    # Validate results
    validation = detector.validate_seal_presence(regions)
    
    print(f"Detection Results:")
    print(f"- Total detections: {len(regions)}")
    print(f"- Saved files: {len(saved_files)}")
    print(f"- Validation: {validation}")


if __name__ == "__main__":
    main()