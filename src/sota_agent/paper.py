import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class ArxivPaper:
    """
    Class for a parsed ArXiv paper from LaTeX source.
    """
    
    # Sections to exclude from LLM processing
    EXCLUDED_SECTIONS = {
        'references', 'bibliography', 'appendix', 'supplementary',
        'acknowledgments', 'acknowledgements', 'author contributions',
        'funding', 'ethics statement', 'checklist', 'supplementary material'
    }
    
    def __init__(self, arxiv_id: str, source_path: Optional[Path] = None, metadata: Optional[Dict] = None):
        """
        Initialize an ArxivPaper instance.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2301.12345")
            source_path: Path to the LaTeX source file
            metadata: Optional metadata dict (title, authors, etc.)
        """
        self.arxiv_id = arxiv_id
        self.source_path = Path(source_path) if source_path else None
        self.metadata = metadata or {}
        self.raw_text = None
        self.sections = []  # List of {"title": str, "content": str, "order": int}
        self.parsed_date = None
        
    def parse(self, latex_text: str):
        """
        Parse raw LaTeX text into structured sections.
        
        Args:
            latex_text: Raw LaTeX source text
        """
        self.raw_text = latex_text
        self.sections = self._parse_latex_sections(latex_text)
        self.parsed_date = datetime.now().isoformat()
    
    def _parse_latex_sections(self, latex_text: str) -> List[Dict]:
        """
        Parse sections from LaTeX source using \\section and \\subsection commands.
        
        Args:
            latex_text: Raw LaTeX text
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Remove comments
        latex_text = self._remove_latex_comments(latex_text)
        
        # Extract abstract first
        abstract_content = self._extract_abstract(latex_text)
        if abstract_content:
            sections.append({
                "title": "Abstract",
                "content": abstract_content,
                "order": 0
            })
        
        # Find all section and subsection commands
        # Pattern matches: \section{title}, \section*{title}, \subsection{title}, etc.
        section_pattern = r'\\(sub)*section\*?\{([^}]+)\}'
        
        matches = list(re.finditer(section_pattern, latex_text))
        
        for i, match in enumerate(matches):
            section_level = match.group(1)  # 'sub' or None
            section_title = match.group(2).strip()
            
            # Clean the title
            section_title = self._clean_latex_title(section_title)
            
            # Skip if invalid
            if not self._is_valid_section_title(section_title):
                continue
            
            # Get content between this section and the next
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(latex_text)
            
            content = latex_text[start_pos:end_pos].strip()
            
            # Clean content
            content = self._clean_latex_content(content)
            
            # Validate content
            if not self._is_valid_section(content):
                continue
            
            # Add section
            sections.append({
                "title": section_title,
                "content": content,
                "order": len(sections)
            })
        
        return sections
    
    def _remove_latex_comments(self, text: str) -> str:
        """Removes LaTeX comments (lines starting with %)."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            # Remove inline comments
            comment_pos = line.find('%')
            if comment_pos >= 0:
                # Check if % is escaped
                if comment_pos == 0 or line[comment_pos - 1] != '\\':
                    line = line[:comment_pos]
            cleaned.append(line)
        return '\n'.join(cleaned)
    
    def _extract_abstract(self, latex_text: str) -> Optional[str]:
        """Extracts content from \\begin{abstract}...\\end{abstract}."""
        pattern = r'\\begin\{abstract\}(.*?)\\end\{abstract\}'
        match = re.search(pattern, latex_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            abstract = match.group(1).strip()
            return self._clean_latex_content(abstract)
        
        return None
    
    def _clean_latex_title(self, title: str) -> str:
        """Cleans LaTeX commands from section title."""
        # Remove common LaTeX commands
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)  # \textbf{text} -> text
        title = re.sub(r'\\[a-zA-Z]+', '', title)  # Remove remaining commands
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        # Remove trailing punctuation
        title = title.rstrip('.:;,')
        
        return title.strip()
    
    def _clean_latex_content(self, content: str) -> str:
        """
        Cleans LaTeX content by removing commands and formatting.
        
        This is a basic cleaner - could be expanded for more sophisticated processing.
        """
        # Remove figure and table environments
        content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
        
        # Remove equation environments (keep inline math for now)
        content = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', content, flags=re.DOTALL)
        
        # Remove citations and references
        content = re.sub(r'\\cite\{[^}]+\}', '', content)
        content = re.sub(r'\\ref\{[^}]+\}', '', content)
        content = re.sub(r'\\label\{[^}]+\}', '', content)
        
        # Remove common formatting commands but keep content
        content = re.sub(r'\\textbf\{([^}]*)\}', r'\1', content)
        content = re.sub(r'\\textit\{([^}]*)\}', r'\1', content)
        content = re.sub(r'\\emph\{([^}]*)\}', r'\1', content)
        
        # Remove remaining LaTeX commands
        content = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', content)
        content = re.sub(r'\\[a-zA-Z]+\*?', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple blank lines
        content = ' '.join(content.split())  # Normalize whitespace
        
        return content.strip()
    
    def _is_valid_section_title(self, title: str) -> bool:
        """Validates if a section title is meaningful."""
        if not title or len(title) < 2:
            return False
        
        # Skip common non-section titles
        skip_patterns = [
            r'^fig',
            r'^figure',
            r'^table',
            r'^appendix\s*[a-z]$',
            r'^[a-z]$',  # Single letter
        ]
        
        title_lower = title.lower()
        for pattern in skip_patterns:
            if re.match(pattern, title_lower):
                return False
        
        return True
    
    def _is_valid_section(self, content: str) -> bool:
        """Validates if section content is meaningful."""
        if not content:
            return False
        
        # Must have minimum length
        if len(content) < 50:
            return False
        
        # Must have at least one sentence
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
        
        if len(meaningful_sentences) < 1:
            return False
        
        return True
    
    def get_relevant_sections(self, exclude_list: Optional[List[str]] = None) -> List[Dict]:
        """
        Get sections that are relevant for LLM analysis (exclude references, appendix, etc.).
        
        Args:
            exclude_list: Optional custom list of section titles to exclude
            
        Returns:
            Filtered list of sections
        """
        if exclude_list is None:
            exclude_set = self.EXCLUDED_SECTIONS
        else:
            exclude_set = set(s.lower() for s in exclude_list)
        
        relevant = []
        for section in self.sections:
            title_lower = section['title'].lower()
            
            # Check if section title contains any excluded keywords
            if not any(excluded in title_lower for excluded in exclude_set):
                relevant.append(section)
        
        return relevant
    
    def get_text_for_llm(self, max_chars: Optional[int] = None, 
                         include_abstract: bool = True) -> str:
        """
        Get concatenated text from relevant sections for LLM processing.
        
        Args:
            max_chars: Maximum characters to return (truncate if exceeded)
            include_abstract: Whether to include abstract from metadata
            
        Returns:
            Formatted text string
        """
        text_parts = []
        
        # Add abstract if available
        if include_abstract and 'abstract' in self.metadata:
            text_parts.append(f"Abstract:\n{self.metadata['abstract']}\n\n")
        
        # Add relevant sections
        relevant_sections = self.get_relevant_sections()
        for section in relevant_sections:
            text_parts.append(f"{section['title']}\n{section['content']}\n\n")
        
        full_text = "".join(text_parts)
        
        # Truncate if needed
        if max_chars and len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[Text truncated...]"
        
        return full_text
    
    def to_dict(self) -> Dict:
        """
        Serialize paper to dictionary for JSON storage.
        
        Returns:
            Dictionary representation
        """
        return {
            "arxiv_id": self.arxiv_id,
            "source_path": str(self.source_path) if self.source_path else None,
            "metadata": self.metadata,
            "sections": self.sections,
            "parsed_date": self.parsed_date
        }
    
    def save_to_json(self, output_path: Path):
        """
        Save parsed paper to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'ArxivPaper':
        """
        Load parsed paper from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            ArxivPaper instance
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paper = cls(
            arxiv_id=data['arxiv_id'],
            source_path=Path(data['source_path']) if data.get('source_path') else None,
            metadata=data.get('metadata', {})
        )
        paper.sections = data.get('sections', [])
        paper.parsed_date = data.get('parsed_date')
        
        return paper
    
    def __repr__(self) -> str:
        return f"ArxivPaper(id={self.arxiv_id}, sections={len(self.sections)})"
