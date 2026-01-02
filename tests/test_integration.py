import json
import pytest
from pathlib import Path


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_data_directory_structure(self):
        """Test that expected data directories exist."""
        base_path = Path(__file__).parent.parent / 'data'
        
        assert base_path.exists(), "Data directory should exist"
        assert (base_path / 'parsed_papers').exists(), "parsed_papers directory should exist"

    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_path = Path(__file__).parent.parent / 'config'
        
        assert config_path.exists(), "Config directory should exist"
        
        # Check for YAML config files
        yaml_files = list(config_path.glob('*.yaml'))
        assert len(yaml_files) > 0, "Should have at least one YAML config file"

    def test_parsed_papers_format(self):
        """Test that parsed papers have expected JSON structure."""
        parsed_path = Path(__file__).parent.parent / 'data' / 'parsed_papers'
        
        if not parsed_path.exists():
            pytest.skip("No parsed papers directory")
        
        json_files = list(parsed_path.glob('*.json'))
        
        if not json_files:
            pytest.skip("No parsed papers found")
        
        # Test first JSON file
        sample_file = json_files[0]
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        # Verify it's a dictionary (basic check)
        assert isinstance(data, dict), "Parsed paper should be a dictionary"

    def test_pyproject_has_dependencies(self):
        """Test that pyproject.toml has required dependencies."""
        pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
        
        assert pyproject_path.exists(), "pyproject.toml should exist"
        
        content = pyproject_path.read_text()
        assert 'pydantic' in content, "Should have pydantic dependency"
        assert 'pytest' in content, "Should have pytest in dev dependencies"

    def test_package_imports(self):
        """Test that main package modules can be imported."""
        try:
            from sota_agent import (
                filter_papers,
                scan_arxiv_metadata,
                download_arxiv_papers,
                analyze_papers
            )
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import package modules: {e}")
