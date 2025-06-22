"""
Utility Functions

This module contains utility functions used across the Documentation Copilot system:
- File validation and handling
- URL validation
- Text processing
- Configuration management
"""

import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
import zipfile
import mimetypes


def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is accessible.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def validate_folder_path(folder_path: str) -> bool:
    """
    Validate if a folder path exists and is accessible.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        True if folder is valid, False otherwise
    """
    try:
        path = Path(folder_path)
        return path.exists() and path.is_dir()
    except Exception:
        return False


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_github_url(url: str) -> bool:
    """
    Validate if a URL is a GitHub repository URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if it's a valid GitHub URL, False otherwise
    """
    if not validate_url(url):
        return False
    
    parsed = urlparse(url)
    return 'github.com' in parsed.netloc


def get_supported_file_extensions() -> List[str]:
    """
    Get list of supported file extensions.
    
    Returns:
        List of supported file extensions
    """
    return ['.md', '.pdf', '.docx', '.txt']


def is_supported_file(file_path: str) -> bool:
    """
    Check if a file is supported by the system.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is supported, False otherwise
    """
    if not validate_file_path(file_path):
        return False
    
    file_extension = Path(file_path).suffix.lower()
    return file_extension in get_supported_file_extensions()


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


def create_temp_directory() -> str:
    """
    Create a temporary directory for file processing.
    
    Returns:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="doc_copilot_")
    return temp_dir


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Clean up a temporary directory.
    
    Args:
        temp_dir: Path to the temporary directory
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")


def extract_zip_file(zip_path: str, extract_to: str) -> List[str]:
    """
    Extract a ZIP file and return list of extracted file paths.
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract to
        
    Returns:
        List of extracted file paths
    """
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
            for root, dirs, files in os.walk(extract_to):
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_supported_file(file_path):
                        extracted_files.append(file_path)
    
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
    
    return extracted_files


def get_file_type_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file type information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {'error': 'File not found'}
    
    file_info = {
        'name': file_path.name,
        'extension': file_path.suffix.lower(),
        'size_mb': get_file_size_mb(str(file_path)),
        'is_supported': is_supported_file(str(file_path)),
        'mime_type': mimetypes.guess_type(str(file_path))[0]
    }
    
    return file_info


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to character count if tiktoken is not available
        return len(text)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content.
    
    Args:
        html_content: HTML content
        
    Returns:
        Clean text content
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return clean_text(text)
    
    except ImportError:
        # Fallback if BeautifulSoup is not available
        import re
        # Simple HTML tag removal
        text = re.sub(r'<[^>]+>', '', html_content)
        return clean_text(text)


def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the current environment.
    
    Returns:
        Dictionary with environment information
    """
    import sys
    import platform
    
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'working_directory': os.getcwd(),
        'environment_variables': {
            'OPENAI_API_KEY': 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT_SET',
            'GROQ_API_KEY': 'SET' if os.getenv('GROQ_API_KEY') else 'NOT_SET',
            'ANTHROPIC_API_KEY': 'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT_SET',
        }
    }


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are set.
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    return {
        'openai': bool(os.getenv('OPENAI_API_KEY')),
        'groq': bool(os.getenv('GROQ_API_KEY')),
        'anthropic': bool(os.getenv('ANTHROPIC_API_KEY'))
    }


def create_sample_documents(output_dir: str = "sample_docs") -> List[str]:
    """
    Create sample documents for testing.
    
    Args:
        output_dir: Directory to create sample documents in
        
    Returns:
        List of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    sample_files = []
    
    # Sample markdown file
    md_content = """# Sample Documentation

This is a sample markdown document for testing the Documentation Copilot.

## Features

- Document processing
- Vector embeddings
- RAG implementation
- Multiple LLM support

## Usage

1. Upload your documents
2. Ask questions
3. Get AI-powered answers

## Configuration

Set your API keys in the environment variables:
- OPENAI_API_KEY
- GROQ_API_KEY
- ANTHROPIC_API_KEY
"""
    
    md_file = output_path / "sample_documentation.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    sample_files.append(str(md_file))
    
    # Sample text file
    txt_content = """Sample Text Document

This is a sample text document that demonstrates the text processing capabilities of the Documentation Copilot.

The system can handle various file formats including:
- Markdown files
- PDF documents
- Word documents
- Plain text files
- Web URLs
- GitHub files

Each document type is processed appropriately to extract meaningful content for the vector database.
"""
    
    txt_file = output_path / "sample_text.txt"
    with open(txt_file, 'w') as f:
        f.write(txt_content)
    sample_files.append(str(txt_file))
    
    return sample_files 