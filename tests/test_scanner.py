import pytest
from unittest.mock import patch

from sota_agent.scanner import scan_arxiv_metadata


@pytest.fixture
def mock_config():
    """Mock configuration for scanner tests."""
    return {
        'max_metadata_scan_limit': 100,
        'keywords': ['machine learning', 'deep learning'],
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'categories': ['cs.AI', 'cs.LG']
    }


@pytest.fixture
def mock_paths(tmp_path):
    """Mock paths using temporary directory."""
    return {
        'DATA': tmp_path / 'arxiv-metadata.json'
    }


@pytest.fixture
def sample_papers():
    """Sample paper metadata."""
    return [
        {
            'id': '2101.00001',
            'title': 'Machine Learning Methods',
            'abstract': 'This paper discusses ML techniques.',
            'categories': 'cs.LG',
            'update_date': '2021-01-01'
        },
        {
            'id': '2101.00002',
            'title': 'Deep Learning for Vision',
            'abstract': 'This paper covers DL in computer vision.',
            'categories': 'cs.CV',
            'update_date': '2021-01-02'
        }
    ]


class TestScanner:
    """Test suite for arXiv metadata scanning."""

    @patch('sota_agent.scanner.stream_arxiv_data')
    @patch('sota_agent.scanner.filter_arxiv_metadata')
    def test_scan_arxiv_metadata_basic(
        self, mock_filter, mock_stream, mock_config, mock_paths, sample_papers
    ):
        """Test basic metadata scanning."""
        mock_stream.return_value = sample_papers
        mock_filter.side_effect = [True, False]  # First matches, second doesn't
        
        result = scan_arxiv_metadata(mock_config, mock_paths)
        
        assert len(result) == 1
        assert result[0]['id'] == '2101.00001'

    @patch('sota_agent.scanner.stream_arxiv_data')
    @patch('sota_agent.scanner.filter_arxiv_metadata')
    def test_scan_respects_max_limit(
        self, mock_filter, mock_stream, mock_config, mock_paths
    ):
        """Test that max_metadata_scan_limit is respected."""
        # Create more papers than the limit
        many_papers = [{'id': f'210{i}'} for i in range(200)]
        mock_stream.return_value = many_papers
        mock_filter.return_value = True
        mock_config['max_metadata_scan_limit'] = 50
        
        result = scan_arxiv_metadata(mock_config, mock_paths)
        
        # Should stop at the limit
        assert len(result) <= 50

    @patch('sota_agent.scanner.stream_arxiv_data')
    def test_scan_with_missing_data_file(self, mock_stream, mock_config, mock_paths):
        """Test handling of missing data file."""
        mock_stream.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(SystemExit):
            scan_arxiv_metadata(mock_config, mock_paths)

    @patch('sota_agent.scanner.stream_arxiv_data')
    @patch('sota_agent.scanner.filter_arxiv_metadata')
    def test_scan_no_candidates_found(
        self, mock_filter, mock_stream, mock_config, mock_paths, sample_papers
    ):
        """Test behavior when no candidates match."""
        mock_stream.return_value = sample_papers
        mock_filter.return_value = False  # Nothing matches
        
        with pytest.raises(SystemExit):
            scan_arxiv_metadata(mock_config, mock_paths)

    @patch('sota_agent.scanner.stream_arxiv_data')
    @patch('sota_agent.scanner.filter_arxiv_metadata')
    def test_scan_unlimited_limit(
        self, mock_filter, mock_stream, mock_config, mock_paths, sample_papers
    ):
        """Test scanning with no limit (-1)."""
        mock_stream.return_value = sample_papers
        mock_filter.return_value = True
        mock_config['max_metadata_scan_limit'] = -1
        
        result = scan_arxiv_metadata(mock_config, mock_paths)
        
        assert len(result) == len(sample_papers)
