import json
import tempfile
from pathlib import Path

from sota_agent.model.pdf_paper import ArxivPdfPaper
from .sample import sample_metadata


class TestArxivPdfPaperInit:
    """Test initialization of ArxivPdfPaper."""
    
    def test_init_with_minimal_args(self):
        """Test creating paper with only required arguments."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345")
        
        assert paper.arxiv_id == "2301.12345"
        assert paper.pdf_path is None
        assert paper.metadata == {}
        assert paper.raw_text is None
        assert paper.gemini_file_uri is None
        assert paper.downloaded_date is None
    
    def test_init_with_metadata(self, sample_metadata):
        """Test creating paper with metadata."""
        paper = ArxivPdfPaper(
            arxiv_id="2301.12345",
            pdf_path=Path("/path/to/paper.pdf"),
            metadata=sample_metadata
        )
        
        assert paper.metadata == sample_metadata
        assert paper.metadata["title"] == "Test Paper"
        assert paper.pdf_path == Path("/path/to/paper.pdf")
    
    def test_init_with_pdf_path_string(self):
        """Test creating paper with PDF path as string."""
        paper = ArxivPdfPaper(
            arxiv_id="2301.12345",
            pdf_path="/path/to/paper.pdf"
        )
        
        assert paper.pdf_path == Path("/path/to/paper.pdf")


class TestGetPdfPath:
    """Test PDF path retrieval."""
    
    def test_get_pdf_path_with_permanent_path(self):
        """Test getting PDF path when permanent path is set."""
        paper = ArxivPdfPaper(
            arxiv_id="2301.12345",
            pdf_path=Path("/path/to/paper.pdf")
        )
        
        assert paper.get_pdf_path_for_upload() == Path("/path/to/paper.pdf")
    
    def test_get_pdf_path_with_temp_path(self):
        """Test getting PDF path when only temp path is set."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345")
        paper._temp_pdf_path = Path("/tmp/paper.pdf")
        
        assert paper.get_pdf_path_for_upload() == Path("/tmp/paper.pdf")
    
    def test_get_pdf_path_permanent_over_temp(self):
        """Test that permanent path takes precedence over temp path."""
        paper = ArxivPdfPaper(
            arxiv_id="2301.12345",
            pdf_path=Path("/path/to/paper.pdf")
        )
        paper._temp_pdf_path = Path("/tmp/paper.pdf")
        
        assert paper.get_pdf_path_for_upload() == Path("/path/to/paper.pdf")


class TestGetRawText:
    """Test raw text retrieval."""
    
    def test_get_raw_text_with_text(self):
        """Test getting raw text when it exists."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345")
        paper.raw_text = "This is extracted text from the PDF."
        
        assert paper.get_raw_text() == "This is extracted text from the PDF."
    
    def test_get_raw_text_when_none(self):
        """Test getting raw text when it's None."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345")
        
        assert paper.get_raw_text() == ""


class TestSerialization:
    """Test JSON serialization and deserialization."""
    
    def test_to_dict(self, sample_metadata):
        """Test converting paper to dictionary."""
        paper = ArxivPdfPaper(
            arxiv_id="2301.12345",
            pdf_path=Path("/path/to/paper.pdf"),
            metadata=sample_metadata
        )
        paper.raw_text = "Extracted text content"
        paper.gemini_file_uri = "gs://bucket/file.pdf"
        paper.downloaded_date = "2025-12-28"
        
        data = paper.to_dict()
        
        assert data['arxiv_id'] == "2301.12345"
        assert data['pdf_path'] == "/path/to/paper.pdf"
        assert data['metadata'] == sample_metadata
        assert data['raw_text'] == "Extracted text content"
        assert data['gemini_file_uri'] == "gs://bucket/file.pdf"
        assert data['downloaded_date'] == "2025-12-28"
    
    def test_to_dict_with_none_values(self):
        """Test converting paper with None values to dictionary."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345")
        
        data = paper.to_dict()
        
        assert data['arxiv_id'] == "2301.12345"
        assert data['pdf_path'] is None
        assert data['metadata'] == {}
        assert data['raw_text'] is None
    
    def test_save_to_json(self, sample_metadata):
        """Test saving paper to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paper = ArxivPdfPaper(
                arxiv_id="2301.12345",
                pdf_path=Path("/path/to/paper.pdf"),
                metadata=sample_metadata
            )
            paper.raw_text = "Extracted text"
            
            output_path = Path(tmpdir) / "test_paper.json"
            paper.save_to_json(output_path)
            
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert data['arxiv_id'] == "2301.12345"
            assert data['raw_text'] == "Extracted text"
    
    def test_from_json(self, sample_metadata):
        """Test loading paper from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a paper
            paper1 = ArxivPdfPaper(
                arxiv_id="2301.12345",
                pdf_path=Path("/path/to/paper.pdf"),
                metadata=sample_metadata
            )
            paper1.raw_text = "Extracted text"
            paper1.gemini_file_uri = "gs://bucket/file.pdf"
            paper1.downloaded_date = "2025-12-28"
            
            json_path = Path(tmpdir) / "test_paper.json"
            paper1.save_to_json(json_path)
            
            # Load it back
            paper2 = ArxivPdfPaper.from_json(json_path)
            
            assert paper2.arxiv_id == paper1.arxiv_id
            assert paper2.pdf_path == paper1.pdf_path
            assert paper2.metadata == paper1.metadata
            assert paper2.raw_text == paper1.raw_text
            assert paper2.gemini_file_uri == paper1.gemini_file_uri
            assert paper2.downloaded_date == paper1.downloaded_date
    
    def test_save_creates_parent_directories(self, sample_metadata):
        """Test that save_to_json creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paper = ArxivPdfPaper(
                arxiv_id="2301.12345",
                pdf_path=Path("/path/to/paper.pdf"),
                metadata=sample_metadata
            )
            
            # Use nested path that doesn't exist
            output_path = Path(tmpdir) / "nested" / "dir" / "paper.json"
            paper.save_to_json(output_path)
            
            assert output_path.exists()


class TestRepr:
    """Test string representation."""
    
    def test_repr_with_pdf(self):
        """Test __repr__ method with PDF path."""
        paper = ArxivPdfPaper(
            arxiv_id="2301.12345",
            pdf_path=Path("/path/to/paper.pdf")
        )
        
        repr_str = repr(paper)
        
        assert "2301.12345" in repr_str
        assert "pdf=Yes" in repr_str
    
    def test_repr_without_pdf(self):
        """Test __repr__ method without PDF path."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345")
        
        repr_str = repr(paper)
        
        assert "2301.12345" in repr_str
        assert "pdf=No" in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_metadata(self):
        """Test paper with empty metadata."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345", metadata={})
        
        assert paper.metadata == {}
    
    def test_arxiv_id_with_version(self):
        """Test ArXiv ID with version suffix."""
        paper = ArxivPdfPaper(arxiv_id="2301.12345v2")
        
        assert paper.arxiv_id == "2301.12345v2"
    
    def test_metadata_with_special_characters(self):
        """Test metadata with special characters."""
        metadata = {
            "title": "Paper with Spëcial Chàracters",
            "abstract": "Contains émojis 🔬 and símbolos"
        }
        paper = ArxivPdfPaper(arxiv_id="2301.12345", metadata=metadata)
        
        assert paper.metadata["title"] == "Paper with Spëcial Chàracters"


# Run tests with: pytest tests/test_arxiv_paper.py -v
