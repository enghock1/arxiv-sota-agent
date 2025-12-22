"""
Integration tests for actual LaTeX source downloading and parsing.

These tests are slower and require network access.

Usage:
    pytest tests/test_fetch_latex.py -v
"""

import tempfile
from pathlib import Path

from sota_agent.paper import ArxivPaper
from sota_agent.utils.fetcher import fetch_arxiv_paper


class TestRealLatexDownload:
    """Test with actual ArXiv LaTeX sources."""
    
    def test_download_and_parse_real_paper(self):
        """Test downloading and parsing a real ArXiv paper from LaTeX."""
        arxiv_id = "1706.03762"  # Attention Is All You Need paper
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Download LaTeX source
            result = fetch_arxiv_paper(arxiv_id, output_dir, output_dir)
            
            # Verify download
            assert result['source_dir'] is not None
            assert result['main_tex'] is not None
            assert result['text'] is not None
            assert len(result['text']) > 1000  # Should have substantial content
            assert result['metadata'] is not None
            
            # Parse with ArxivPaper class
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                source_path=Path(result['main_tex']),
                metadata=result['metadata']
            )
            paper.parse(result['text'])
            
            # Verify parsing
            assert len(paper.sections) > 0
            
            # Check for expected sections in this paper
            section_titles = [s['title'].lower() for s in paper.sections]
            assert any('introduction' in title for title in section_titles)
    
    def test_get_llm_text_from_real_latex(self):
        """Test getting LLM-ready text from real LaTeX source."""
        arxiv_id = "1706.03762"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = fetch_arxiv_paper(arxiv_id, output_dir, output_dir)
            
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                source_path=Path(result['main_tex']) if result['main_tex'] else None,
                metadata=result['metadata']
            )
            paper.parse(result['text'])
            
            # Get text for LLM with limit
            llm_text = paper.get_text_for_llm(max_chars=10000)
            
            assert len(llm_text) <= 10100  # Allow buffer
            assert len(llm_text) > 0
            
            # Should exclude references
            relevant = paper.get_relevant_sections()
            relevant_titles = [s['title'].lower() for s in relevant]
            assert not any(title == 'references' for title in relevant_titles)
    
    def test_save_and_load_parsed_real_paper(self):
        """Test saving and loading a parsed real paper."""
        arxiv_id = "1706.03762"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = fetch_arxiv_paper(arxiv_id, output_dir, output_dir)
            
            # Parse and save
            paper1 = ArxivPaper(
                arxiv_id=arxiv_id,
                source_path=Path(result['main_tex']) if result['main_tex'] else None,
                metadata=result['metadata']
            )
            paper1.parse(result['text'])
            
            json_path = output_dir / f"{arxiv_id}.json"
            paper1.save_to_json(json_path)
            
            # Load back
            paper2 = ArxivPaper.from_json(json_path)
            
            # Verify same data
            assert paper2.arxiv_id == paper1.arxiv_id
            assert len(paper2.sections) == len(paper1.sections)


class TestLatexParsingQuality:
    """Test quality of LaTeX text extraction and parsing."""
    
    def test_nested_section_detection(self):
        """Test if subsections are detected properly."""
        arxiv_id = "1706.03762"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = fetch_arxiv_paper(arxiv_id, output_dir, output_dir)
            
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                source_path=Path(result['main_tex']) if result['main_tex'] else None
            )
            paper.parse(result['text'])
            
            section_titles = [s['title'] for s in paper.sections]
            
            # LaTeX parser should detect sections and subsections
            print(f"Number of sections: {len(section_titles)}")
            print(f"Section titles: {section_titles[:5]}")
            
            # Transformer paper should have multiple sections
            assert len(section_titles) > 10
    
    def test_special_characters_in_real_latex(self):
        """Test handling of special characters from real LaTeX."""
        arxiv_id = "1706.03762"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = fetch_arxiv_paper(arxiv_id, output_dir, output_dir)
            
            # Should not crash on any special characters
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                source_path=Path(result['main_tex']) if result['main_tex'] else None
            )
            paper.parse(result['text'])
            
            assert len(paper.sections) > 0
