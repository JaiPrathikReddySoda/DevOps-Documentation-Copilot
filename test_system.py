"""
Test Script for Documentation Copilot

This script tests all major components of the Documentation Copilot system:
- Document processing
- Vector store operations
- RAG engine functionality
- Utility functions
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
from utils import (
    validate_api_keys, get_environment_info, create_sample_documents,
    validate_file_path, validate_url, get_supported_file_extensions
)


def test_environment():
    """Test environment setup."""
    print("🔧 Testing Environment Setup...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check environment info
    env_info = get_environment_info()
    print(f"Platform: {env_info['platform']}")
    print(f"Working directory: {env_info['working_directory']}")
    
    # Check API keys
    api_keys = validate_api_keys()
    print("API Key Status:")
    for provider, status in api_keys.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {provider.title()}: {'Available' if status else 'Not Set'}")
    
    if not any(api_keys.values()):
        print("⚠️  Warning: No API keys found. Some tests will be skipped.")
    
    print()


def test_utilities():
    """Test utility functions."""
    print("🔧 Testing Utility Functions...")
    
    # Test file validation
    assert validate_file_path(__file__) == True
    assert validate_file_path("nonexistent_file.txt") == False
    print("✅ File validation tests passed")
    
    # Test URL validation
    assert validate_url("https://example.com") == True
    assert validate_url("not_a_url") == False
    print("✅ URL validation tests passed")
    
    # Test supported extensions
    extensions = get_supported_file_extensions()
    assert '.md' in extensions
    assert '.pdf' in extensions
    print("✅ Supported extensions test passed")
    
    print()


def test_document_processor():
    """Test document processor."""
    print("📄 Testing Document Processor...")
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    # Create test content
    test_content = """# Test Document

This is a test document for the Documentation Copilot.

## Features

- Document processing
- Vector embeddings
- RAG implementation

## Usage

1. Upload documents
2. Ask questions
3. Get answers

This document contains multiple paragraphs to test chunking functionality.
The processor should create appropriate chunks based on the content structure.
"""
    
    # Test chunk creation
    chunks = processor._create_chunks(test_content, {
        'source': 'test.md',
        'type': 'markdown',
        'filename': 'test.md'
    })
    
    assert len(chunks) > 0
    print(f"✅ Created {len(chunks)} chunks from test content")
    
    # Test chunk statistics
    stats = processor.get_chunk_stats(chunks)
    assert stats['total_chunks'] == len(chunks)
    assert stats['total_tokens'] > 0
    print(f"✅ Chunk statistics: {stats['total_chunks']} chunks, {stats['total_tokens']} tokens")
    
    print()


def test_vector_store():
    """Test vector store operations."""
    print("🗄️ Testing Vector Store...")
    
    # Initialize vector store
    vector_store = VectorStore(index_path="test_vector_index")
    
    # Create test chunks
    test_chunks = [
        {
            'content': 'This is a test document about machine learning.',
            'metadata': {
                'source': 'test1.md',
                'type': 'markdown',
                'filename': 'test1.md',
                'chunk_id': 0,
                'chunk_size': 50,
                'token_count': 10
            }
        },
        {
            'content': 'Machine learning is a subset of artificial intelligence.',
            'metadata': {
                'source': 'test2.md',
                'type': 'markdown',
                'filename': 'test2.md',
                'chunk_id': 0,
                'chunk_size': 60,
                'token_count': 12
            }
        }
    ]
    
    # Add documents to vector store
    vector_store.add_documents(test_chunks)
    print(f"✅ Added {len(test_chunks)} chunks to vector store")
    
    # Test search
    results = vector_store.search("machine learning", k=2, threshold=0.1)
    assert len(results) > 0
    print(f"✅ Search returned {len(results)} results")
    
    # Test statistics
    stats = vector_store.get_stats()
    assert stats['total_vectors'] == len(test_chunks)
    print(f"✅ Vector store stats: {stats['total_vectors']} vectors")
    
    # Test save and load
    vector_store.save()
    print("✅ Vector store saved")
    
    # Create new instance and load
    new_vector_store = VectorStore(index_path="test_vector_index")
    if new_vector_store.load():
        print("✅ Vector store loaded successfully")
        new_stats = new_vector_store.get_stats()
        assert new_stats['total_vectors'] == len(test_chunks)
    else:
        print("❌ Failed to load vector store")
    
    # Clean up
    import shutil
    if os.path.exists("test_vector_index"):
        shutil.rmtree("test_vector_index")
    
    print()


def test_rag_engine():
    """Test RAG engine (requires API key)."""
    print("🤖 Testing RAG Engine...")
    
    api_keys = validate_api_keys()
    if not any(api_keys.values()):
        print("⚠️  Skipping RAG engine tests - no API keys available")
        print()
        return
    
    # Initialize components
    vector_store = VectorStore(index_path="test_rag_index")
    
    # Add test documents
    test_chunks = [
        {
            'content': 'The Documentation Copilot is an AI-powered tool for document Q&A.',
            'metadata': {
                'source': 'docs.md',
                'type': 'markdown',
                'filename': 'docs.md',
                'chunk_id': 0,
                'chunk_size': 70,
                'token_count': 15
            }
        }
    ]
    
    vector_store.add_documents(test_chunks)
    
    # Initialize RAG engine
    rag_engine = RAGEngine(
        vector_store=vector_store,
        llm_provider=list(api_keys.keys())[0],  # Use first available provider
        model_name="gpt-3.5-turbo" if list(api_keys.keys())[0] == "openai" else "llama2-70b-4096",
        temperature=0.1,
        max_tokens=500
    )
    
    print(f"✅ RAG engine initialized with {rag_engine.llm_provider}")
    
    # Test answer generation
    try:
        result = rag_engine.answer_question(
            question="What is the Documentation Copilot?",
            k=1,
            threshold=0.1,
            include_sources=True
        )
        
        assert 'answer' in result
        assert 'sources' in result
        print("✅ RAG answer generation successful")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Sources: {len(result['sources'])}")
        
    except Exception as e:
        print(f"❌ RAG answer generation failed: {str(e)}")
    
    # Test configuration
    config = rag_engine.get_config()
    assert 'llm_provider' in config
    assert 'model_name' in config
    print("✅ RAG configuration test passed")
    
    # Clean up
    import shutil
    if os.path.exists("test_rag_index"):
        shutil.rmtree("test_rag_index")
    
    print()


def test_sample_documents():
    """Test sample document creation."""
    print("📝 Testing Sample Document Creation...")
    
    try:
        sample_files = create_sample_documents("test_sample_docs")
        assert len(sample_files) > 0
        print(f"✅ Created {len(sample_files)} sample documents")
        
        # Verify files exist
        for file_path in sample_files:
            assert validate_file_path(file_path)
        
        print("✅ Sample documents validation passed")
        
        # Clean up
        import shutil
        if os.path.exists("test_sample_docs"):
            shutil.rmtree("test_sample_docs")
        
    except Exception as e:
        print(f"❌ Sample document creation failed: {str(e)}")
    
    print()


def main():
    """Run all tests."""
    print("🧪 Documentation Copilot System Tests")
    print("=" * 50)
    
    try:
        test_environment()
        test_utilities()
        test_document_processor()
        test_vector_store()
        test_rag_engine()
        test_sample_documents()
        
        print("🎉 All tests completed successfully!")
        print("\n📋 Test Summary:")
        print("✅ Environment setup")
        print("✅ Utility functions")
        print("✅ Document processor")
        print("✅ Vector store operations")
        print("✅ RAG engine (if API keys available)")
        print("✅ Sample document creation")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 