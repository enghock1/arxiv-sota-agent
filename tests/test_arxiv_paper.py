import json
import tempfile
from pathlib import Path

from sota_agent.paper import ArxivPaper
from .sample import (sample_metadata, sample_latex_text, sample_latex_with_subsections, sample_latex_minimal)


class TestArxivPaperInit:
    """Test initialization of ArxivPaper."""
    
    def test_init_with_minimal_args(self):
        """Test creating paper with only required arguments."""
        paper = ArxivPaper(arxiv_id="2301.12345")
        
        assert paper.arxiv_id == "2301.12345"
        assert paper.source_path is None
        assert paper.metadata == {}
        assert paper.raw_text is None
        assert paper.sections == []
        assert paper.parsed_date is None
    
    def test_init_with_metadata(self, sample_metadata):
        """Test creating paper with metadata."""
        paper = ArxivPaper(
            arxiv_id="2301.12345",
            source_path=Path("/path/to/source.tex"),
            metadata=sample_metadata
        )
        
        assert paper.metadata == sample_metadata
        assert paper.metadata["title"] == "Test Paper"


class TestParseSections:
    """Test section parsing functionality."""
    
    def test_parse_latex_sections(self, sample_latex_text):
        """Test parsing LaTeX text with sections."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_text)
        
        assert len(paper.sections) > 0
        assert paper.raw_text == sample_latex_text
        assert paper.parsed_date is not None
        
        # Check that some expected sections are present
        section_titles = [s['title'] for s in paper.sections]
        assert any('Introduction' in title for title in section_titles)
        assert any('Conclusion' in title for title in section_titles)
    
    def test_parse_latex_with_subsections(self, sample_latex_with_subsections):
        """Test parsing LaTeX with subsections."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_with_subsections)
        
        # Should parse sections and subsections
        assert len(paper.sections) >= 1
        assert paper.raw_text == sample_latex_with_subsections
    
    def test_parse_minimal_latex(self, sample_latex_minimal):
        """Test parsing minimal LaTeX document."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_minimal)
        
        # Should parse abstract and sections
        assert len(paper.sections) >= 1
        assert paper.raw_text == sample_latex_minimal
    
    def test_parse_no_clear_sections(self):
        """Test parsing text with no clear section headers."""
        paper = ArxivPaper(arxiv_id="test")
        text = "This is just plain text without any LaTeX section commands at all."
        paper.parse(text)
        
        # May not find sections without \section{} commands
        assert len(paper.sections) >= 0
    
    def test_section_order_preserved(self, sample_latex_text):
        """Test that section order is preserved."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_text)
        
        orders = [s['order'] for s in paper.sections]
        assert orders == sorted(orders)  # Should be in ascending order


class TestGetRelevantSections:
    """Test filtering of relevant sections."""
    
    def test_exclude_references_section(self, sample_latex_text):
        """Test that references/bibliography sections are excluded."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_text)
        
        relevant = paper.get_relevant_sections()
        relevant_titles = [s['title'].lower() for s in relevant]
        
        # Should not have references or bibliography
        assert 'references' not in relevant_titles
        assert 'bibliography' not in relevant_titles
    
    def test_exclude_appendix(self, sample_latex_minimal):
        """Test that appendix is excluded."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_minimal)
        
        relevant = paper.get_relevant_sections()
        relevant_titles = [s['title'].lower() for s in relevant]
        
        assert not any('appendix' in title for title in relevant_titles)
    
    def test_custom_exclude_list(self, sample_latex_text):
        """Test using custom exclude list."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_text)
        
        relevant = paper.get_relevant_sections(exclude_list=['introduction', 'conclusion'])
        relevant_titles = [s['title'].lower() for s in relevant]
        
        assert not any('introduction' in title for title in relevant_titles)
        assert not any('conclusion' in title for title in relevant_titles)
    
    def test_keep_method_sections(self, sample_latex_text):
        """Test that important sections are kept."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_text)
        
        relevant = paper.get_relevant_sections()
        
        # Should have some relevant sections
        assert len(relevant) > 0


class TestGetTextForLLM:
    """Test LLM text generation functionality."""
    
    def test_get_text_basic(self, sample_latex_text, sample_metadata):
        """Test getting text for LLM with basic options."""
        paper = ArxivPaper(arxiv_id="test", metadata=sample_metadata)
        paper.parse(sample_latex_text)
        
        text = paper.get_text_for_llm()
        
        assert len(text) > 0
        assert sample_metadata['abstract'] in text  # Abstract should be included
    
    def test_get_text_without_abstract(self, sample_latex_text, sample_metadata):
        """Test getting text without abstract."""
        paper = ArxivPaper(arxiv_id="test", metadata=sample_metadata)
        paper.parse(sample_latex_text)
        
        text = paper.get_text_for_llm(include_abstract=False)
        
        assert sample_metadata['abstract'] not in text
    
    def test_get_text_with_max_chars(self, sample_latex_text):
        """Test text truncation with max_chars."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_text)
        
        max_chars = 200
        text = paper.get_text_for_llm(max_chars=max_chars)
        
        assert len(text) <= max_chars + 50  # Allow some buffer for truncation message
        assert "[Text truncated...]" in text
    
    def test_get_text_no_metadata_abstract(self, sample_latex_text):
        """Test getting text when metadata has no abstract."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse(sample_latex_text)
        
        text = paper.get_text_for_llm(include_abstract=True)
        
        # Should not crash, just skip abstract
        assert len(text) > 0


class TestSerialization:
    """Test JSON serialization and deserialization."""
    
    def test_to_dict(self, sample_metadata, sample_latex_text):
        """Test converting paper to dictionary."""
        paper = ArxivPaper(
            arxiv_id="2301.12345",
            source_path=Path("/path/to/source.tex"),
            metadata=sample_metadata
        )
        paper.parse(sample_latex_text)
        
        data = paper.to_dict()
        
        assert data['arxiv_id'] == "2301.12345"
        assert data['source_path'] == "/path/to/source.tex"
        assert data['metadata'] == sample_metadata
        assert len(data['sections']) > 0
        assert data['parsed_date'] is not None
    
    def test_save_to_json(self, sample_metadata, sample_latex_text):
        """Test saving paper to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paper = ArxivPaper(
                arxiv_id="2301.12345",
                source_path=Path("/path/to/source.tex"),
                metadata=sample_metadata
            )
            paper.parse(sample_latex_text)
            
            output_path = Path(tmpdir) / "test_paper.json"
            paper.save_to_json(output_path)
            
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert data['arxiv_id'] == "2301.12345"
            assert len(data['sections']) > 0
    
    def test_from_json(self, sample_metadata, sample_latex_text):
        """Test loading paper from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a paper
            paper1 = ArxivPaper(
                arxiv_id="2301.12345",
                source_path=Path("/path/to/source.tex"),
                metadata=sample_metadata
            )
            paper1.parse(sample_latex_text)
            
            json_path = Path(tmpdir) / "test_paper.json"
            paper1.save_to_json(json_path)
            
            # Load it back
            paper2 = ArxivPaper.from_json(json_path)
            
            assert paper2.arxiv_id == paper1.arxiv_id
            assert paper2.source_path == paper1.source_path
            assert paper2.metadata == paper1.metadata
            assert len(paper2.sections) == len(paper1.sections)
            assert paper2.parsed_date == paper1.parsed_date
    
    def test_save_creates_parent_directories(self, sample_metadata):
        """Test that save_to_json creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paper = ArxivPaper(
                arxiv_id="2301.12345",
                source_path=Path("/path/to/source.tex"),
                metadata=sample_metadata
            )
            paper.parse("Some text")
            
            # Use nested path that doesn't exist
            output_path = Path(tmpdir) / "nested" / "dir" / "paper.json"
            paper.save_to_json(output_path)
            
            assert output_path.exists()


class TestRepr:
    """Test string representation."""
    
    def test_repr(self, sample_latex_text):
        """Test __repr__ method."""
        paper = ArxivPaper(arxiv_id="2301.12345")
        paper.parse(sample_latex_text)
        
        repr_str = repr(paper)
        
        assert "2301.12345" in repr_str
        assert "sections" in repr_str.lower()
        assert str(len(paper.sections)) in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self):
        """Test parsing empty text."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse("")
        
        # Should handle gracefully
        assert len(paper.sections) >= 0
    
    def test_very_short_text(self):
        """Test parsing very short text."""
        paper = ArxivPaper(arxiv_id="test")
        paper.parse("Short")
        
        # Should not crash
        assert paper.raw_text == "Short"
    
    def test_special_characters_in_text(self):
        """Test parsing text with special characters."""
        paper = ArxivPaper(arxiv_id="test")
        text = r"\section{Introducción}" + "\nContenido con caracteres especiales: é, ñ, ü"
        paper.parse(text)
        
        # Should handle special characters
        assert paper.raw_text == text
    
    def test_multiple_parse_calls(self, sample_latex_text):
        """Test calling parse multiple times."""
        paper = ArxivPaper(arxiv_id="test")
        
        paper.parse("First text")
        first_date = paper.parsed_date
        
        paper.parse(sample_latex_text)
        second_date = paper.parsed_date
        
        # Should update with new data
        assert paper.raw_text == sample_latex_text
        assert first_date != second_date


# Run tests with: pytest tests/test_arxiv_paper.py -v
