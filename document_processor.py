"""
Document Processor Module

This module handles the parsing and chunking of various document formats:
- Markdown files (.md)
- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)
- Web URLs
- GitHub files

The processor creates intelligent chunks that preserve context and are optimal for vector embedding.
"""

import os
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import markdown
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Main document processor class that handles multiple file formats
    and creates intelligent chunks for vector embedding.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Initialize tokenizer for length calculation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single file and return chunks with metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of dictionaries containing chunks with metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and process accordingly
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.md':
            return self._process_markdown(file_path)
        elif file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension == '.docx':
            return self._process_docx(file_path)
        elif file_extension == '.txt':
            return self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def process_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Process content from a web URL.
        
        Args:
            url: URL to fetch and process
            
        Returns:
            List of dictionaries containing chunks with metadata
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return self._create_chunks(text, {
                'source': url,
                'type': 'web_url',
                'title': soup.title.string if soup.title else url
            })
            
        except Exception as e:
            raise Exception(f"Failed to process URL {url}: {str(e)}")
    
    def process_github_file(self, github_url: str) -> List[Dict[str, Any]]:
        """
        Process a file from a GitHub repository URL.
        
        Args:
            github_url: Full URL to the file on GitHub
            
        Returns:
            List of dictionaries containing chunks with metadata
        """
        # Regex to extract user, repo, branch, and file path from various GitHub URL formats
        match = re.match(r"https://github\.com/([^/]+)/([^/]+)/(?:blob|tree)/([^/]+)/(.+)", github_url)
        if not match:
            raise ValueError("Invalid GitHub file URL format. Please provide a URL that includes the branch, like '/blob/main/' or '/tree/main/'.")

        user, repo, branch, file_path = match.groups()
        repo_url_base = f"https://github.com/{user}/{repo}"
        
        # Construct the raw content URL
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
        
        try:
            response = requests.get(raw_url, timeout=10)
            response.raise_for_status()
            
            content = response.text
            
            return self._create_chunks(content, {
                'source': github_url,
                'type': 'github_file',
                'repository': repo_url_base,
                'file_path': file_path
            })
            
        except Exception as e:
            raise Exception(f"Failed to process GitHub file {raw_url}: {str(e)}")
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported files in a folder recursively.
        
        Args:
            folder_path: Path to the folder to process
            
        Returns:
            List of dictionaries containing chunks with metadata
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        all_chunks = []
        supported_extensions = {'.md', '.pdf', '.docx', '.txt'}
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    chunks = self.process_file(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {str(e)}")
                    continue
        
        return all_chunks
    
    def _process_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from markdown frontmatter if present
        metadata = self._extract_markdown_metadata(content)
        
        # Convert markdown to plain text for chunking
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        return self._create_chunks(text, {
            'source': str(file_path),
            'type': 'markdown',
            'filename': file_path.name,
            **metadata
        })
    
    def _process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a PDF file."""
        chunks = []
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():
                    page_chunks = self._create_chunks(text, {
                        'source': str(file_path),
                        'type': 'pdf',
                        'filename': file_path.name,
                        'page': page_num + 1,
                        'total_pages': len(pdf_reader.pages)
                    })
                    chunks.extend(page_chunks)
        
        return chunks
    
    def _process_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a Word document."""
        doc = Document(file_path)
        
        # Extract text from paragraphs
        text_content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        text = '\n\n'.join(text_content)
        
        return self._create_chunks(text, {
            'source': str(file_path),
            'type': 'docx',
            'filename': file_path.name
        })
    
    def _process_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self._create_chunks(content, {
            'source': str(file_path),
            'type': 'text',
            'filename': file_path.name
        })
    
    def _extract_markdown_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from markdown frontmatter."""
        metadata = {}
        
        # Check for YAML frontmatter
        if content.startswith('---'):
            try:
                end_index = content.find('---', 3)
                if end_index != -1:
                    frontmatter = content[3:end_index].strip()
                    # Simple YAML parsing (for MVP purposes)
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip().strip('"\'')
            except:
                pass
        
        return metadata
    
    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks from text with metadata.
        
        Args:
            text: Text content to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk_text),
                'token_count': len(self.tokenizer.encode(chunk_text))
            })
            
            chunks.append({
                'content': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the processed chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_tokens': 0,
                'avg_chunk_size': 0,
                'file_types': {},
                'sources': set()
            }
        
        total_tokens = sum(chunk['metadata']['token_count'] for chunk in chunks)
        file_types = {}
        sources = set()
        
        for chunk in chunks:
            file_type = chunk['metadata']['type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
            sources.add(chunk['metadata']['source'])
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_chunk_size': total_tokens / len(chunks),
            'file_types': file_types,
            'sources': list(sources)
        } 