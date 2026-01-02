import pytest
from unittest.mock import Mock, patch

from sota_agent.arxiv_download import download_arxiv_papers
from sota_agent.model.pdf_paper import ArxivPdfPaper


@pytest.fixture
def mock_config():
    """Mock configuration for download tests."""
    return {
        'max_download_calls': 5,
        'save_files': False,
        'save_parsed_papers': True
    }


@pytest.fixture
def mock_paths(tmp_path):
    """Mock paths using temporary directory."""
    return {
        'PARSED_PAPERS': tmp_path / 'parsed_papers',
        'SOURCES': tmp_path / 'sources'
    }


@pytest.fixture
def sample_candidates():
    """Sample candidate paper metadata."""
    return [
        {'id': '2101.00001', 'title': 'Test Paper 1', 'authors': 'Author A'},
        {'id': '2101.00002', 'title': 'Test Paper 2', 'authors': 'Author B'},
    ]


class TestArxivDownload:
    """Test suite for arXiv download functionality."""

    @patch('sota_agent.arxiv_download.fetch_paper_from_arxiv')
    def test_download_arxiv_papers_creates_directories(
        self, mock_fetch, mock_config, mock_paths, sample_candidates
    ):
        """Test that required directories are created."""
        mock_fetch.return_value = Mock(spec=ArxivPdfPaper)
        
        download_arxiv_papers(mock_config, sample_candidates, mock_paths)
        
        assert mock_paths['PARSED_PAPERS'].exists()

    @patch('sota_agent.arxiv_download.fetch_paper_from_arxiv')
    def test_download_respects_max_calls(
        self, mock_fetch, mock_config, mock_paths, sample_candidates
    ):
        """Test that max_download_calls is respected."""
        mock_fetch.return_value = Mock(spec=ArxivPdfPaper)
        mock_config['max_download_calls'] = 1
        
        download_arxiv_papers(mock_config, sample_candidates, mock_paths)
        
        # Should only call fetch once despite having 2 candidates
        assert mock_fetch.call_count <= 1

    @patch('sota_agent.arxiv_download.fetch_paper_from_arxiv')
    def test_download_with_no_limit(
        self, mock_fetch, mock_config, mock_paths, sample_candidates
    ):
        """Test download with no limit (-1)."""
        mock_fetch.return_value = Mock(spec=ArxivPdfPaper)
        mock_config['max_download_calls'] = -1
        
        download_arxiv_papers(mock_config, sample_candidates, mock_paths)
        
        # Should call fetch for all candidates
        assert mock_fetch.call_count == len(sample_candidates)

    def test_download_with_empty_candidates(self, mock_config, mock_paths):
        """Test download with empty candidate list."""
        result = download_arxiv_papers(mock_config, [], mock_paths)
        assert result == []
