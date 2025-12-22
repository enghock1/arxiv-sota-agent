import gzip
import tarfile
import requests
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any


def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches metadata for an ArXiv paper using the ArXiv API.
    
    Args:
        arxiv_id: ArXiv paper ID (e.g., "2301.12345" or "1706.03762")
        
    Returns:
        Dictionary with metadata (title, authors, abstract, published, updated, categories)
        or None if fetch failed
    """
    # Clean the arxiv_id
    arxiv_id = arxiv_id.replace('arxiv:', '').strip()
    
    # ArXiv API URL
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        # Find the entry (paper)
        entry = root.find('atom:entry', ns)
        if entry is None:
            print(f"No entry found for ArXiv ID: {arxiv_id}")
            return None
        
        # Extract metadata
        metadata = {}
        
        # Title
        title = entry.find('atom:title', ns)
        if title is not None and title.text is not None:
            metadata['title'] = ' '.join(title.text.strip().split())
        
        # Authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None and name.text is not None:
                authors.append(name.text.strip())
        metadata['authors'] = authors
        
        # Abstract
        abstract = entry.find('atom:summary', ns)
        if abstract is not None and abstract.text is not None:
            metadata['abstract'] = ' '.join(abstract.text.strip().split())
        
        # Published date
        published = entry.find('atom:published', ns)
        if published is not None and published.text is not None:
            metadata['published'] = published.text.strip()
        
        # Updated date
        updated = entry.find('atom:updated', ns)
        if updated is not None and updated.text is not None:
            metadata['updated'] = updated.text.strip()
        
        # Primary category
        primary_category = entry.find('arxiv:primary_category', ns)
        if primary_category is not None:
            metadata['primary_category'] = primary_category.get('term')
        
        # All categories
        categories = []
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term:
                categories.append(term)
        metadata['categories'] = categories
        
        # DOI if available
        doi = entry.find('arxiv:doi', ns)
        if doi is not None and doi.text is not None:
            metadata['doi'] = doi.text.strip()
        
        # Journal reference if available
        journal_ref = entry.find('arxiv:journal_ref', ns)
        if journal_ref is not None and journal_ref.text is not None:
            metadata['journal_ref'] = journal_ref.text.strip()
        
        return metadata
        
    except Exception as e:
        print(f"Failed to fetch metadata for {arxiv_id}: {e}")
        return None


def download_arxiv_source(arxiv_id: str, output_dir: Path, timeout: int = 30) -> Optional[Path]:
    """
    Downloads LaTeX source files from ArXiv.
    
    Args:
        arxiv_id: ArXiv paper ID (e.g., "2301.12345")
        output_dir: Directory to save the source files
        timeout: Request timeout in seconds
        
    Returns:
        Path to extracted source directory, or None if download failed
    """
    # Clean the arxiv_id
    arxiv_id = arxiv_id.replace('arxiv:', '').strip()
    
    # Construct source URL
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / arxiv_id
    
    # Skip if already downloaded
    if source_dir.exists() and any(source_dir.iterdir()):
        print(f"Source already exists: {source_dir}")
        return source_dir
    
    try:
        # print(f"Downloading source: {source_url}")
        response = requests.get(source_url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Extract the archive
        source_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try as tar.gz first
            with tarfile.open(tmp_path, 'r:gz') as tar:
                tar.extractall(source_dir)
        except tarfile.ReadError:
            # Try as gzip (single .tex file)
            with gzip.open(tmp_path, 'rb') as gz:
                content = gz.read()
                # Save as main.tex
                (source_dir / 'main.tex').write_bytes(content)
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        # print(f"Extracted to: {source_dir}")
        return source_dir
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download source for {arxiv_id}: {e}")
        return None
    except Exception as e:
        print(f"Failed to extract source for {arxiv_id}: {e}")
        return None


def find_main_tex_file(source_dir: Path) -> Optional[Path]:
    """
    Finds the main .tex file in a source directory.
    
    Args:
        source_dir: Directory containing LaTeX source files
        
    Returns:
        Path to main .tex file, or None if not found
    """
    # Look for common main file names
    common_names = ['main.tex', 'paper.tex', 'manuscript.tex', 'article.tex']
    
    for name in common_names:
        tex_file = source_dir / name
        if tex_file.exists():
            return tex_file
    
    # Find .tex files
    tex_files = list(source_dir.glob('*.tex'))
    
    if not tex_files:
        return None
    
    # If only one .tex file, use it
    if len(tex_files) == 1:
        return tex_files[0]
    
    # Look for file with \documentclass or \begin{document}
    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding='utf-8', errors='ignore')
            if r'\documentclass' in content or r'\begin{document}' in content:
                return tex_file
        except Exception:
            continue
    
    # Return the largest .tex file as fallback
    return max(tex_files, key=lambda f: f.stat().st_size)


def extract_text_from_latex(tex_file: Path) -> Optional[str]:
    """
    Extracts text content from a LaTeX file, resolving \\input{} includes.
    
    Args:
        tex_file: Path to the .tex file
        
    Returns:
        Raw LaTeX text as string with includes resolved, or None if extraction failed
    """
    try:
        # Read main file with multiple encoding attempts
        text = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                text = tex_file.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            # If all fail, read with errors='ignore'
            text = tex_file.read_text(encoding='utf-8', errors='ignore')
        
        # Resolve \input{file} commands
        text = _resolve_latex_inputs(text, tex_file.parent)
        
        return text
        
    except Exception as e:
        print(f"Failed to read LaTeX file {tex_file}: {e}")
        return None


def _resolve_latex_inputs(text: str, base_dir: Path) -> str:
    """
    Recursively resolves \\input{file} commands in LaTeX text.
    
    Args:
        text: LaTeX text
        base_dir: Base directory for resolving relative paths
        
    Returns:
        LaTeX text with all inputs resolved
    """
    import re
    
    # Pattern for \input{filename} (no extension or .tex extension)
    input_pattern = r'\\input\{([^}]+)\}'
    
    def replace_input(match):
        filename = match.group(1)
        if filename:
            filename = filename.strip()
        else:
            return match.group(0)
        
        # Try with .tex extension if not provided
        if not filename.endswith('.tex'):
            tex_path = base_dir / f"{filename}.tex"
        else:
            tex_path = base_dir / filename
        
        if not tex_path.exists():
            # Try without extension
            tex_path = base_dir / filename
        
        if not tex_path.exists():
            # Can't find file, return original command
            print(f"Warning: Could not find input file: {filename}")
            return match.group(0)
        
        try:
            # Read the included file
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    included_text = tex_path.read_text(encoding=encoding)
                    # Recursively resolve inputs in the included file
                    included_text = _resolve_latex_inputs(included_text, base_dir)
                    return included_text
                except UnicodeDecodeError:
                    continue
            
            # Fallback with errors='ignore'
            included_text = tex_path.read_text(encoding='utf-8', errors='ignore')
            included_text = _resolve_latex_inputs(included_text, base_dir)
            return included_text
            
        except Exception as e:
            print(f"Warning: Failed to read input file {tex_path}: {e}")
            return match.group(0)
    
    # Replace all \input commands
    resolved_text = re.sub(input_pattern, replace_input, text)
    
    return resolved_text


def fetch_arxiv_paper(arxiv_id: str, parsed_papers_dir: Path, output_dir: Path, keep_source: bool = True) -> dict:
    """
    Downloads ArXiv paper source and metadata.
    
    Args:
        arxiv_id: ArXiv paper ID
        parsed_papers_dir: Directory of parsed papers
        output_dir: Directory to save source files (used only if keep_source=True)
        keep_source: If True, saves source to output_dir. If False, uses temp directory and deletes after extraction.
        
    Returns:
        Dictionary with 'source_dir', 'main_tex', 'text', and 'metadata'
    """
    result = {
        'arxiv_id': arxiv_id,
        'source_dir': None,
        'main_tex': None,
        'text': None,
        'metadata': None
    }

    # print("Fetching metadata...")
    metadata = fetch_arxiv_metadata(arxiv_id)
    result['metadata'] = metadata
    
    # Determine download directory
    if keep_source:
        download_dir = output_dir
        cleanup_dir = None
    else:
        # Use temporary directory that will be cleaned up
        download_dir = Path(tempfile.mkdtemp())
        cleanup_dir = download_dir
    
    try:
        # Download source
        source_dir = download_arxiv_source(arxiv_id, download_dir)
        if not source_dir:
            return result
        
        result['source_dir'] = str(source_dir) if keep_source else None
        
        # Find main .tex file
        main_tex = find_main_tex_file(source_dir)
        if not main_tex:
            print(f"No .tex file found in {source_dir}")
            return result
        
        result['main_tex'] = str(main_tex) if keep_source else None
        # print(f"Found main file: {main_tex.name}")
        
        # Extract text
        text = extract_text_from_latex(main_tex)
        result['text'] = text
        
    finally:
        # Clean up temp directory if needed
        if cleanup_dir and cleanup_dir.exists():
            import shutil
            shutil.rmtree(cleanup_dir)
            # print("Cleaned up temporary source files")
    
    return result
