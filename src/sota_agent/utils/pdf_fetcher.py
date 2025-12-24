import PyPDF2
import requests
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from sota_agent.pdf_paper import ArxivPdfPaper


def download_arxiv_pdf(arxiv_id: str, output_dir: Path, timeout: int = 30) -> Optional[Path]:
    """
    Downloads PDF from ArXiv.
    
    Args:
        arxiv_id: ArXiv paper ID (e.g., "2301.12345")
        output_dir: Directory to save the PDF file
        timeout: Request timeout in seconds
        
    Returns:
        Path to downloaded PDF, or None if download failed
    """
    # Clean the arxiv_id
    arxiv_id = arxiv_id.replace('arxiv:', '').strip()
    
    # Construct PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{arxiv_id}.pdf"
    
    # Skip if already downloaded
    if pdf_path.exists():
        return pdf_path
    
    try:
        response = requests.get(pdf_url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Save PDF
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return pdf_path
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF for {arxiv_id}: {e}")
        return None


def extract_text_from_pdf_for_filtering(pdf_path: Path, max_pages: int = 10) -> str:
    """
    Quickly extract text from PDF for keyword filtering.
    Only extracts first N pages for efficiency.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract (default: 10)
        
    Returns:
        Extracted text as string
    """
    if not pdf_path.exists():
        return ""
    
    try:
        text_parts = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            pages_to_extract = min(max_pages, num_pages)
            
            for page_num in range(pages_to_extract):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    # Skip problematic pages
                    continue
        
        return " ".join(text_parts)
        
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def fetch_arxiv_pdf_paper(
    arxiv_id: str, 
    paper_metadata: Dict[str, Any],
    pdf_dir: Path, 
    keep_pdf: bool = True
) -> Optional['ArxivPdfPaper']:
    """
    Downloads ArXiv paper PDF and creates ArxivPdfPaper object.
    
    Args:
        arxiv_id: ArXiv paper ID
        paper_metadata: Metadata dict from arxiv dataset scan
        pdf_dir: Directory to save PDF files
        keep_pdf: If True, keeps PDF. If False, uses temp directory
        
    Returns:
        ArxivPdfPaper object or None if download failed
    """
    
    # Determine download directory
    if keep_pdf:
        download_dir = pdf_dir
        cleanup_path = None
    else:
        # Use temporary directory
        download_dir = Path(tempfile.mkdtemp())
        cleanup_path = download_dir
    
    try:
        # Download PDF
        pdf_path = download_arxiv_pdf(arxiv_id, download_dir)
        if not pdf_path:
            return None
        
        # Create ArxivPdfPaper object
        pdf_paper = ArxivPdfPaper(
            arxiv_id=arxiv_id,
            pdf_path=pdf_path if keep_pdf else None,
            metadata=paper_metadata
        )
        
        # Extract text for filtering (even if not keeping PDF)
        pdf_paper.raw_text = extract_text_from_pdf_for_filtering(pdf_path, max_pages=10)
        
        # If not keeping PDF, store temporary path for later Gemini upload
        if not keep_pdf:
            pdf_paper._temp_pdf_path = pdf_path
        
        return pdf_paper
        
    except Exception as e:
        print(f"Failed to process PDF for {arxiv_id}: {e}")
        return None
    finally:
        # Clean up temp directory only after we're done with the PDF
        # (Actual cleanup happens after Gemini upload)
        pass
