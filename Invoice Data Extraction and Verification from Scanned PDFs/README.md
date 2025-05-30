# Invoice Extraction System (Work in Progress)

This is a modular Python project to extract structured invoice data from scanned PDFs. The system is currently under development and focuses on accurate extraction of fields using OCR and preprocessing techniques.

---

## Current Status

The following components have been implemented and tested:

### Implemented Modules
| Module | Description |
|--------|-------------|
| `pdf_processor.py` | Converts multi-page PDF invoices to separate image files (PNG/JPEG). |
| `image_preprocessor.py` | Preprocesses images for better OCR performance (e.g., denoising, binarization). |
| `ocr_engine.py` | Performs OCR using Tesseract and returns text with bounding boxes. |
| `field_extractor.py` | Extracts fields like Invoice Number, Dates, GST, PO Number, etc., using regex and spatial grouping. |
| `main.py` | Orchestrates the pipeline from PDF to extracted field dictionary. |


---

## Directory Structure (Working as of Now)

invoice_extraction/
├── input/ # Input folder containing scanned PDF invoices
│ └── invoice_sample.pdf
├── output/ # Outputs are stored here
│ └── extracted_data.json # Final extracted field dictionary
├── src/
│ ├── main.py # Main pipeline runner
│ ├── modules/
│ │ ├── pdf_processor.py
│ │ ├── image_preprocessor.py
│ │ ├── ocr_engine.py
│ │ └── field_extractor.py
│ └── utils/
│ ├── config.py # Configuration parameters
│ └── logger.py # Basic logging
├── requirements.txt
└── README.md


---

##  Input Format

- Drop scanned invoice PDFs into the `input/` directory.
- Each file should be named meaningfully (e.g., `invoice_sample.pdf`).

---

## Output Format (Current)

- **JSON** file saved to `output/extracted_data.json`
- Example:

```json
{
  "invoice_number": "INV-2023-001",
  "invoice_date": "2023-04-15",
  "due_date": "2023-05-15",
  "po_number": "PO-78452",
  "gst_number": "29ABCDE1234F2Z5",
  "shipping_address": "123 Tech Park, Bengaluru, KA, India"
}

## How to Run
1. Clone the repository:
git clone https://github.com/yourname/invoice_extraction.git
cd invoice_extraction

2. Set up a virtual environment:
python -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Place PDF files in input/ and run the pipeline:
python src/main.py

## Configuration
Edit src/utils/config.py to control:
OCR engine (currently Tesseract only)
Enable/disable preprocessing steps
Paths for input/output
Field regex patterns

## Next Modules (Planned)
Table Parsing
Signature/Seal Detection
Field Confidence & Verifiability Scoring
Output to Excel
Unit Testing





