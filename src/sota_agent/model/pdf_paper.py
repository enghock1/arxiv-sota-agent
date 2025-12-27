import json
from pathlib import Path
from typing import Optional, Dict


class ArxivPdfPaper:
    """
    Class for an ArXiv paper processed via PDF (no section parsing).
    Used as alternative to LaTeX-based ArxivPaper when full document analysis is needed.
    """
    
    def __init__(self, arxiv_id: str, pdf_path: Optional[Path] = None, metadata: Optional[Dict] = None):
        """
        Initialize an ArxivPdfPaper instance.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            pdf_path: Path to the PDF file (can be None if using temp storage)
            metadata: Optional metadata dict (title, authors, abstract, etc.)
        """
        self.arxiv_id = arxiv_id
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.metadata = metadata or {}
        self.raw_text: Optional[str] = None  # Extracted text for filtering (first 10 pages)
        self.gemini_file_uri: Optional[str] = None  # Cached URI after upload to Gemini
        self.downloaded_date: Optional[str] = None
        self._temp_pdf_path: Optional[Path] = None  # Temporary path if not keeping PDF
    
    def get_pdf_path_for_upload(self) -> Optional[Path]:
        """
        Get the PDF path to use for Gemini upload.
        Returns either the permanent path or temporary path.
        """
        return self.pdf_path if self.pdf_path else self._temp_pdf_path
    
    def get_raw_text(self) -> str:
        """
        Get cached raw text.
        
        Returns:
            Extracted text string
        """
        return self.raw_text if self.raw_text else ""
    
    def upload_to_gemini(self, client):
        """
        Uploads PDF to Gemini File API and caches the file object.
        
        Args:
            client: Gemini client instance with file upload capability
            
        Returns:
            Uploaded file object for use in Gemini API calls
        """
        if self.gemini_file_uri:
            # Return cached file object (stored as URI, but in new API we need to re-upload or store object)
            # For now, we'll upload each time if no cached object
            pass
        
        # Get PDF path (permanent or temporary)
        pdf_path = self.get_pdf_path_for_upload()
        if not pdf_path or not pdf_path.exists():
            raise ValueError(f"PDF file not found for {self.arxiv_id}: {pdf_path}")
        
        # Upload to Gemini File API (Google AI SDK)
        uploaded_file = client.files.upload(file=str(pdf_path))
        
        # Cache the URI for reference
        if hasattr(uploaded_file, 'uri'):
            self.gemini_file_uri = uploaded_file.uri
        
        return uploaded_file
    
    def to_dict(self) -> Dict:
        """
        Serialize paper to dictionary for JSON storage.
        
        Returns:
            Dictionary representation
        """
        return {
            "arxiv_id": self.arxiv_id,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "metadata": self.metadata,
            "raw_text": self.raw_text,
            "gemini_file_uri": self.gemini_file_uri,
            "downloaded_date": self.downloaded_date
        }
    
    def save_to_json(self, output_path: Path):
        """
        Save PDF paper metadata to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'ArxivPdfPaper':
        """
        Load PDF paper from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            ArxivPdfPaper instance
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paper = cls(
            arxiv_id=data['arxiv_id'],
            pdf_path=Path(data['pdf_path']) if data.get('pdf_path') else None,
            metadata=data.get('metadata', {})
        )
        paper.raw_text = data.get('raw_text')
        paper.gemini_file_uri = data.get('gemini_file_uri')
        paper.downloaded_date = data.get('downloaded_date')
        
        return paper
    
    def __repr__(self) -> str:
        return f"ArxivPdfPaper(id={self.arxiv_id}, pdf={'Yes' if self.pdf_path else 'No'})"
