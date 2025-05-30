"""
Module 5: Table Parser (table_parser.py)
Purpose: Extract and parse line item tables from invoices
"""

import re
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class LineItem:
    """Data class to hold line item information"""
    item_description: str
    hsn_code: str
    quantity: float
    unit_price: float
    total_price: float
    unit: str
    discount: float = 0.0
    tax_rate: float = 0.0
    tax_amount: float = 0.0
    confidence: float = 0.0
    coordinates: Tuple[int, int, int, int] = (0, 0, 0, 0)

@dataclass
class TableRegion:
    """Data class to hold table region information"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    rows: List[List[str]]

class TableParser:
    """
    Extracts and parses line item tables from invoices
    """
    
    def __init__(self):
        """Initialize table parser with configurations"""
        self.table_headers = [
            'description', 'item', 'product', 'service',
            'hsn', 'sac', 'code',
            'qty', 'quantity', 'qnty',
            'rate', 'price', 'unit price', 'amount',
            'total', 'value', 'sum'
        ]
        
        self.unit_keywords = [
            'pcs', 'nos', 'kg', 'gm', 'ltr', 'mtr', 'sqft', 'box', 'pack'
        ]
        
        # Table detection parameters
        self.min_table_rows = 2
        self.min_table_cols = 3
        self.cell_min_width = 50
        self.cell_min_height = 20
        
    def detect_table_regions(self, image: np.ndarray, ocr_results: List[Dict]) -> List[TableRegion]:
        """
        Identify table boundaries in the image
        
        Args:
            image (np.ndarray): Invoice image
            ocr_results (List[Dict]): OCR results with coordinates
            
        Returns:
            List[TableRegion]: Detected table regions
        """
        try:
            table_regions = []
            
            # Method 1: OCR-based table detection
            ocr_tables = self._detect_tables_from_ocr(ocr_results)
            table_regions.extend(ocr_tables)
            
            # Method 2: Line-based table detection
            if len(table_regions) == 0:
                line_tables = self._detect_tables_from_lines(image, ocr_results)
                table_regions.extend(line_tables)
            
            # Method 3: Fallback - grid-based detection
            if len(table_regions) == 0:
                grid_tables = self._detect_tables_from_grid(image, ocr_results)
                table_regions.extend(grid_tables)
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Error detecting table regions: {str(e)}")
            return []
    
    def extract_table_rows(self, table_region: TableRegion, ocr_results: List[Dict]) -> List[List[str]]:
        """
        Parse individual table rows from a table region
        
        Args:
            table_region (TableRegion): Detected table region
            ocr_results (List[Dict]): OCR results with coordinates
            
        Returns:
            List[List[str]]: Parsed table rows
        """
        try:
            # Filter OCR results within table region
            table_texts = []
            for result in ocr_results:
                bbox = result.get('bbox', (0, 0, 0, 0))
                if self._is_within_region(bbox, table_region):
                    table_texts.append({
                        'text': result['text'],
                        'bbox': bbox,
                        'confidence': result.get('confidence', 0.0)
                    })
            
            # Group texts by rows based on Y coordinates
            rows = self._group_texts_by_rows(table_texts)
            
            # Sort rows by Y coordinate
            sorted_rows = sorted(rows, key=lambda x: x['y_center'])
            
            # Extract text from each row and organize into columns
            table_rows = []
            for row in sorted_rows:
                row_texts = sorted(row['texts'], key=lambda x: x['bbox'][0])  # Sort by X coordinate
                row_data = [text['text'] for text in row_texts]
                if row_data:  # Only add non-empty rows
                    table_rows.append(row_data)
            
            return table_rows
            
        except Exception as e:
            logger.error(f"Error extracting table rows: {str(e)}")
            return []
    
    def parse_line_items(self, table_data: List[List[str]]) -> List[LineItem]:
        """
        Structure line item data from parsed table
        
        Args:
            table_data (List[List[str]]): Raw table data
            
        Returns:
            List[LineItem]: Structured line items
        """
        try:
            if not table_data or len(table_data) < 2:
                logger.warning("Insufficient table data for line item parsing")
                return []
            
            # Identify header row and column mapping
            header_mapping = self._identify_columns(table_data[0])
            
            line_items = []
            
            # Process data rows (skip header)
            for row_idx, row in enumerate(table_data[1:], 1):
                if len(row) < 3:  # Skip rows with insufficient data
                    continue
                
                try:
                    line_item = self._extract_line_item_from_row(row, header_mapping)
                    if line_item:
                        line_items.append(line_item)
                except Exception as e:
                    logger.warning(f"Error parsing row {row_idx}: {str(e)}")
                    continue
            
            return line_items
            
        except Exception as e:
            logger.error(f"Error parsing line items: {str(e)}")
            return []
    
    def extract_item_details(self, row_data: List[str], header_mapping: Dict[str, int]) -> Optional[LineItem]:
        """
        Extract detailed item information from a single row
        
        Args:
            row_data (List[str]): Single row data
            header_mapping (Dict[str, int]): Column index mapping
            
        Returns:
            Optional[LineItem]: Extracted line item
        """
        try:
            return self._extract_line_item_from_row(row_data, header_mapping)
            
        except Exception as e:
            logger.error(f"Error extracting item details: {str(e)}")
            return None
    
    def _detect_tables_from_ocr(self, ocr_results: List[Dict]) -> List[TableRegion]:
        """Detect tables based on OCR text patterns"""
        try:
            table_regions = []
            
            # Look for table header indicators
            header_candidates = []
            for result in ocr_results:
                text_lower = result['text'].lower()
                if any(header in text_lower for header in self.table_headers):
                    header_candidates.append(result)
            
            if not header_candidates:
                return []
            
            # Group nearby headers to form table regions
            for header in header_candidates:
                bbox = header['bbox']
                
                # Find nearby texts that could be table data
                nearby_texts = []
                for result in ocr_results:
                    other_bbox = result['bbox']
                    if (abs(other_bbox[1] - bbox[3]) < 200 and  # Within reasonable vertical distance
                        abs(other_bbox[0] - bbox[0]) < 500):    # Within reasonable horizontal distance
                        nearby_texts.append(result)
                
                if len(nearby_texts) >= self.min_table_rows:
                    # Calculate table region boundaries
                    all_boxes = [header['bbox']] + [t['bbox'] for t in nearby_texts]
                    x1 = min(box[0] for box in all_boxes)
                    y1 = min(box[1] for box in all_boxes)
                    x2 = max(box[2] for box in all_boxes)
                    y2 = max(box[3] for box in all_boxes)
                    
                    table_regions.append(TableRegion(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=0.8,
                        rows=[]
                    ))
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Error in OCR-based table detection: {str(e)}")
            return []
    
    def _detect_tables_from_lines(self, image: np.ndarray, ocr_results: List[Dict]) -> List[TableRegion]:
        """Detect tables based on line detection"""
        try:
            table_regions = []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w > 200 and h > 100:  # Minimum table size
                    # Check if region contains OCR text
                    texts_in_region = [r for r in ocr_results 
                                     if self._is_within_region(r['bbox'], 
                                                             TableRegion(x, y, x+w, y+h, 0.0, []))]
                    
                    if len(texts_in_region) >= self.min_table_rows:
                        table_regions.append(TableRegion(
                            x1=x, y1=y, x2=x+w, y2=y+h,
                            confidence=0.7,
                            rows=[]
                        ))
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Error in line-based table detection: {str(e)}")
            return []
    
    def _detect_tables_from_grid(self, image: np.ndarray, ocr_results: List[Dict]) -> List[TableRegion]:
        """Fallback grid-based table detection"""
        try:
            table_regions = []
            
            if not ocr_results:
                return []
            
            # Create a simple grid based on OCR text positions
            y_positions = sorted(set(result['bbox'][1] for result in ocr_results))
            
            # Group texts by approximate rows
            row_groups = []
            current_row = []
            current_y = y_positions[0]
            tolerance = 20  # Y-coordinate tolerance
            
            for result in sorted(ocr_results, key=lambda x: x['bbox'][1]):
                if abs(result['bbox'][1] - current_y) <= tolerance:
                    current_row.append(result)
                else:
                    if len(current_row) >= self.min_table_cols:
                        row_groups.append(current_row)
                    current_row = [result]
                    current_y = result['bbox'][1]
            
            # Add last row if valid
            if len(current_row) >= self.min_table_cols:
                row_groups.append(current_row)
            
            # Create table region if we have enough rows
            if len(row_groups) >= self.min_table_rows:
                all_boxes = [result['bbox'] for row in row_groups for result in row]
                x1 = min(box[0] for box in all_boxes)
                y1 = min(box[1] for box in all_boxes)
                x2 = max(box[2] for box in all_boxes)
                y2 = max(box[3] for box in all_boxes)
                
                table_regions.append(TableRegion(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=0.6,
                    rows=[]
                ))
            
            return table_regions
            
        except Exception as e:
            logger.error(f"Error in grid-based table detection: {str(e)}")
            return []
    
    def _is_within_region(self, bbox: Tuple[int, int, int, int], region: TableRegion) -> bool:
        """Check if bounding box is within table region"""
        try:
            x1, y1, x2, y2 = bbox
            return (x1 >= region.x1 and y1 >= region.y1 and 
                   x2 <= region.x2 and y2 <= region.y2)
        except:
            return False
    
    def _group_texts_by_rows(self, texts: List[Dict]) -> List[Dict]:
        """Group OCR texts by rows based on Y coordinates"""
        try:
            if not texts:
                return []
            
            # Sort by Y coordinate
            sorted_texts = sorted(texts, key=lambda x: x['bbox'][1])
            
            rows = []
            current_row = {'y_center': sorted_texts[0]['bbox'][1], 'texts': [sorted_texts[0]]}
            tolerance = 15  # Y-coordinate tolerance for same row
            
            for text in sorted_texts[1:]:
                text_y = text['bbox'][1]
                
                if abs(text_y - current_row['y_center']) <= tolerance:
                    current_row['texts'].append(text)
                else:
                    rows.append(current_row)
                    current_row = {'y_center': text_y, 'texts': [text]}
            
            # Add last row
            rows.append(current_row)
            
            return rows
            
        except Exception as e:
            logger.error(f"Error grouping texts by rows: {str(e)}")
            return []
    
    def _identify_columns(self, header_row: List[str]) -> Dict[str, int]:
        """Identify column types from header row"""
        try:
            mapping = {}
            
            for idx, header in enumerate(header_row):
                header_lower = header.lower().strip()
                
                # Description/Item column
                if any(keyword in header_lower for keyword in ['description', 'item', 'product', 'service']):
                    mapping['description'] = idx
                
                # HSN/SAC code column
                elif any(keyword in header_lower for keyword in ['hsn', 'sac', 'code']):
                    mapping['hsn'] = idx
                
                # Quantity column
                elif any(keyword in header_lower for keyword in ['qty', 'quantity', 'qnty']):
                    mapping['quantity'] = idx
                
                # Unit price column
                elif any(keyword in header_lower for keyword in ['rate', 'price', 'unit']):
                    mapping['unit_price'] = idx
                
                # Total/Amount column
                elif any(keyword in header_lower for keyword in ['total', 'amount', 'value']):
                    mapping['total'] = idx
                
                # Unit column
                elif any(keyword in header_lower for keyword in ['unit', 'uom']):
                    mapping['unit'] = idx
                
                # Tax/GST column
                elif any(keyword in header_lower for keyword in ['tax', 'gst', 'igst', 'cgst', 'sgst']):
                    mapping['tax'] = idx
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error identifying columns: {str(e)}")
            return {}
    
    def _extract_line_item_from_row(self, row: List[str], header_mapping: Dict[str, int]) -> Optional[LineItem]:
        """Extract line item from a single row"""
        try:
            if len(row) < 3:
                return None
            
            # Initialize with defaults
            item = LineItem(
                item_description="",
                hsn_code="",
                quantity=0.0,
                unit_price=0.0,
                total_price=0.0,
                unit="",
                confidence=0.5
            )
            
            # Extract fields based on mapping
            if 'description' in header_mapping and header_mapping['description'] < len(row):
                item.item_description = row[header_mapping['description']].strip()
            
            if 'hsn' in header_mapping and header_mapping['hsn'] < len(row):
                item.hsn_code = row[header_mapping['hsn']].strip()
            
            if 'quantity' in header_mapping and header_mapping['quantity'] < len(row):
                item.quantity = self._parse_number(row[header_mapping['quantity']])
            
            if 'unit_price' in header_mapping and header_mapping['unit_price'] < len(row):
                item.unit_price = self._parse_number(row[header_mapping['unit_price']])
            
            if 'total' in header_mapping and header_mapping['total'] < len(row):
                item.total_price = self._parse_number(row[header_mapping['total']])
            
            if 'unit' in header_mapping and header_mapping['unit'] < len(row):
                item.unit = row[header_mapping['unit']].strip()
            
            # Fallback: try to extract from row based on patterns if mapping failed
            if not item.item_description:
                item.item_description = self._find_description_in_row(row)
            
            if item.quantity == 0.0:
                item.quantity = self._find_quantity_in_row(row)
            
            if item.unit_price == 0.0:
                item.unit_price = self._find_price_in_row(row)
            
            if item.total_price == 0.0:
                item.total_price = self._find_total_in_row(row)
            
            # Validate that we have minimum required fields
            if (item.item_description and 
                (item.quantity > 0 or item.unit_price > 0 or item.total_price > 0)):
                return item
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting line item from row: {str(e)}")
            return None
    
    def _parse_number(self, text: str) -> float:
        """Parse number from text string"""
        try:
            if not text:
                return 0.0
            
            # Remove currency symbols and spaces
            cleaned = re.sub(r'[₹$€£,\s]', '', text.strip())
            
            # Extract number
            match = re.search(r'(\d+\.?\d*)', cleaned)
            if match:
                return float(match.group(1))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error parsing number from '{text}': {str(e)}")
            return 0.0
    
    def _find_description_in_row(self, row: List[str]) -> str:
        """Find item description in row (usually longest text)"""
        try:
            if not row:
                return ""
            
            # Find the longest non-numeric string
            description_candidates = []
            for cell in row:
                cell = cell.strip()
                if cell and not re.match(r'^\d+\.?\d*$', cell):
                    description_candidates.append(cell)
            
            if description_candidates:
                return max(description_candidates, key=len)
            
            return row[0] if row else ""
            
        except Exception as e:
            logger.error(f"Error finding description in row: {str(e)}")
            return ""
    
    def _find_quantity_in_row(self, row: List[str]) -> float:
        """Find quantity in row"""
        try:
            for cell in row:
                # Look for small numbers (likely quantities)
                num = self._parse_number(cell)
                if 0 < num <= 1000:  # Reasonable quantity range
                    return num
            return 0.0
        except:
            return 0.0
    
    def _find_price_in_row(self, row: List[str]) -> float:
        """Find unit price in row"""
        try:
            prices = []
            for cell in row:
                num = self._parse_number(cell)
                if num > 0:
                    prices.append(num)
            
            if len(prices) >= 2:
                # Unit price is typically the smaller amount
                return min(prices)
            elif prices:
                return prices[0]
            
            return 0.0
        except:
            return 0.0
    
    def _find_total_in_row(self, row: List[str]) -> float:
        """Find total amount in row"""
        try:
            prices = []
            for cell in row:
                num = self._parse_number(cell)
                if num > 0:
                    prices.append(num)
            
            if prices:
                # Total is typically the largest amount
                return max(prices)
            
            return 0.0
        except:
            return 0.0


# Example usage and testing
def test_table_parser():
    """Test function for table parser"""
    
    # Mock OCR results representing a table
    mock_ocr_results = [
        {'text': 'Description', 'bbox': (50, 100, 150, 120), 'confidence': 0.9},
        {'text': 'HSN', 'bbox': (200, 100, 250, 120), 'confidence': 0.9},
        {'text': 'Qty', 'bbox': (300, 100, 350, 120), 'confidence': 0.9},
        {'text': 'Rate', 'bbox': (400, 100, 450, 120), 'confidence': 0.9},
        {'text': 'Amount', 'bbox': (500, 100, 570, 120), 'confidence': 0.9},
        
        {'text': 'Laptop Computer', 'bbox': (50, 130, 180, 150), 'confidence': 0.85},
        {'text': '8471', 'bbox': (200, 130, 240, 150), 'confidence': 0.8},
        {'text': '2', 'bbox': (300, 130, 320, 150), 'confidence': 0.9},
        {'text': '45000', 'bbox': (400, 130, 450, 150), 'confidence': 0.88},
        {'text': '90000', 'bbox': (500, 130, 550, 150), 'confidence': 0.9},
        
        {'text': 'Software License', 'bbox': (50, 160, 170, 180), 'confidence': 0.82},
        {'text': '9983', 'bbox': (200, 160, 240, 180), 'confidence': 0.75},
        {'text': '1', 'bbox': (300, 160, 320, 180), 'confidence': 0.9},
        {'text': '15000', 'bbox': (400, 160, 450, 180), 'confidence': 0.85},
        {'text': '15000', 'bbox': (500, 160, 550, 180), 'confidence': 0.88},
    ]
    
    # Create mock image
    mock_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    parser = TableParser()
    
    print("=== Table Parser Test ===")
    
    # Detect table regions
    table_regions = parser.detect_table_regions(mock_image, mock_ocr_results)
    print(f"Detected {len(table_regions)} table region(s)")
    
    if table_regions:
        # Extract table rows
        table_rows = parser.extract_table_rows(table_regions[0], mock_ocr_results)
        print(f"Extracted {len(table_rows)} rows")
        
        for i, row in enumerate(table_rows):
            print(f"Row {i}: {row}")
        
        # Parse line items
        line_items = parser.parse_line_items(table_rows)
        print(f"\nParsed {len(line_items)} line items:")
        
        for i, item in enumerate(line_items):
            print(f"Item {i+1}:")
            print(f"  Description: {item.item_description}")
            print(f"  HSN: {item.hsn_code}")
            print(f"  Quantity: {item.quantity}")
            print(f"  Unit Price: {item.unit_price}")
            print(f"  Total: {item.total_price}")
            print(f"  Confidence: {item.confidence:.2f}")


if __name__ == "__main__":
    test_table_parser()