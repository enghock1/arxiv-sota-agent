import pytest
from unittest.mock import Mock

from sota_agent.filter import filter_papers
from sota_agent.model.pdf_paper import ArxivPdfPaper


@pytest.fixture
def mock_config():
    """Mock configuration for filter tests."""
    return {
        'content_keywords': ['machine learning', 'neural network'],
        'preview_filtered_papers': False
    }


@pytest.fixture
def mock_paths(tmp_path):
    """Mock paths using temporary directory."""
    return {
        'OUTPUT': tmp_path / 'output'
    }


@pytest.fixture
def sample_pdf_papers():
    """Create sample PDF paper objects."""
    paper1 = Mock(spec=ArxivPdfPaper)
    paper1.get_raw_text.return_value = "This paper discusses machine learning techniques."
    paper1.to_dict.return_value = {'id': '1', 'title': 'ML Paper'}
    
    paper2 = Mock(spec=ArxivPdfPaper)
    paper2.get_raw_text.return_value = "This paper is about quantum computing."
    paper2.to_dict.return_value = {'id': '2', 'title': 'QC Paper'}
    
    paper3 = Mock(spec=ArxivPdfPaper)
    paper3.get_raw_text.return_value = "Deep neural network architectures are explored."
    paper3.to_dict.return_value = {'id': '3', 'title': 'NN Paper'}
    
    return [paper1, paper2, paper3]


class TestFilter:
    """Test suite for paper filtering functionality."""

    def test_filter_papers_with_keywords(self, mock_config, mock_paths, sample_pdf_papers):
        """Test filtering papers by content keywords."""
        result = filter_papers(mock_config, sample_pdf_papers, mock_paths)
        
        # Should match paper1 (machine learning) and paper3 (neural network)
        assert len(result) == 2

    def test_filter_papers_no_keywords(self, mock_paths, sample_pdf_papers):
        """Test filtering with no keywords returns all papers."""
        config = {'content_keywords': []}
        
        result = filter_papers(config, sample_pdf_papers, mock_paths)
        
        assert len(result) == len(sample_pdf_papers)

    def test_filter_papers_case_insensitive(self, mock_paths, sample_pdf_papers):
        """Test that keyword matching is case-insensitive."""
        config = {'content_keywords': ['MACHINE LEARNING']}
        
        result = filter_papers(config, sample_pdf_papers, mock_paths)
        
        assert len(result) >= 1

    def test_filter_papers_no_matches(self, mock_paths, sample_pdf_papers):
        """Test filtering with keywords that match nothing."""
        config = {'content_keywords': ['nonexistent_keyword_xyz']}
        
        with pytest.raises(SystemExit):
            filter_papers(config, sample_pdf_papers, mock_paths)

    def test_filter_papers_with_preview(self, mock_paths, sample_pdf_papers):
        """Test filtering with preview enabled."""
        config = {
            'content_keywords': ['machine learning', 'neural network'],
            'preview_filtered_papers': True
        }
        
        result = filter_papers(config, sample_pdf_papers, mock_paths)
        
        # Check that preview file would be created
        preview_path = mock_paths['OUTPUT'] / "filtered_papers_preview.json"
        assert preview_path.exists()
