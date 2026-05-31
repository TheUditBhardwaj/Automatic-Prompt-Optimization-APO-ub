import pdfplumber
import os
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text content from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file on disk.
        
    Returns:
        A single merged string containing all extracted text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                else:
                    logger.warning(f"No text extracted from page {page_num} of {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to parse PDF {pdf_path}: {e}")
        raise RuntimeError(f"Error reading PDF {pdf_path}: {e}")
        
    return "\n\n".join(text_content).strip()

if __name__ == "__main__":
    # Quick self-test block
    import sys
    if len(sys.argv) > 1:
        logging.basicConfig(level=logging.INFO)
        test_file = sys.argv[1]
        print(f"Extracting from: {test_file}")
        try:
            txt = extract_text_from_pdf(test_file)
            print(f"Extracted {len(txt)} chars.")
            print("Prefix:")
            print(txt[:200])
        except Exception as ex:
            print(f"Error: {ex}")
