"""
Quick Start Script for Documentation Copilot

This script helps you get started with the Documentation Copilot MVP:
1. Sets up the environment
2. Creates sample documents
3. Processes and embeds the documents
4. Demonstrates the RAG functionality
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
from utils import validate_api_keys, create_sample_documents


def check_environment():
    """Check if the environment is properly set up."""
    print("🔧 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check API keys
    api_keys = validate_api_keys()
    available_providers = [k for k, v in api_keys.items() if v]
    
    if not available_providers:
        print("⚠️  No API keys found. You can still test document processing and vector storage.")
        print("   To use RAG functionality, set one of: OPENAI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY")
        return False
    
    print(f"✅ API keys available for: {', '.join(available_providers)}")
    return True


def create_sample_data():
    """Create sample documents for testing."""
    print("\n📝 Creating sample documents...")
    
    try:
        sample_files = create_sample_documents("sample_docs")
        print(f"✅ Created {len(sample_files)} sample documents:")
        for file_path in sample_files:
            print(f"   - {Path(file_path).name}")
        return sample_files
    except Exception as e:
        print(f"❌ Failed to create sample documents: {str(e)}")
        return []


def process_documents(sample_files):
    """Process the sample documents."""
    print("\n🔄 Processing documents...")
    
    try:
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        
        for file_path in sample_files:
            print(f"   Processing {Path(file_path).name}...")
            chunks = processor.process_file(file_path)
            all_chunks.extend(chunks)
            print(f"   ✅ Created {len(chunks)} chunks")
        
        print(f"✅ Total chunks created: {len(all_chunks)}")
        return all_chunks
        
    except Exception as e:
        print(f"❌ Failed to process documents: {str(e)}")
        return []


def setup_vector_store(chunks):
    """Set up the vector store with processed chunks."""
    print("\n🗄️ Setting up vector store...")
    
    try:
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        vector_store.save()
        
        stats = vector_store.get_stats()
        print(f"✅ Vector store created with {stats['total_vectors']} vectors")
        print(f"   Sources: {stats['unique_sources']}")
        print(f"   File types: {list(stats['file_types'].keys())}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Failed to set up vector store: {str(e)}")
        return None


def test_rag_functionality(vector_store):
    """Test the RAG functionality."""
    print("\n🤖 Testing RAG functionality...")
    
    api_keys = validate_api_keys()
    if not any(api_keys.values()):
        print("⚠️  Skipping RAG test - no API keys available")
        return None
    
    try:
        # Initialize RAG engine
        provider = list(api_keys.keys())[0]
        model = "gpt-3.5-turbo" if provider == "openai" else "llama2-70b-4096"
        
        rag_engine = RAGEngine(
            vector_store=vector_store,
            llm_provider=provider,
            model_name=model,
            temperature=0.1,
            max_tokens=500
        )
        
        print(f"✅ RAG engine initialized with {provider}")
        
        # Test questions
        test_questions = [
            "What is the Documentation Copilot?",
            "What file types are supported?",
            "How do I use the system?"
        ]
        
        print("\n🧪 Testing sample questions:")
        for question in test_questions:
            print(f"\nQ: {question}")
            try:
                result = rag_engine.answer_question(question, k=3, threshold=0.3)
                print(f"A: {result['answer'][:150]}...")
                print(f"   Sources: {len(result['sources'])}")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        return rag_engine
        
    except Exception as e:
        print(f"❌ Failed to test RAG functionality: {str(e)}")
        return None


def show_next_steps():
    """Show next steps for the user."""
    print("\n🎉 Quick start completed!")
    print("\n📋 Next Steps:")
    print("1. 🚀 Start the web interface:")
    print("   streamlit run app.py")
    print("\n2. 🤖 Start the Slack bot (optional):")
    print("   python slack_bot.py")
    print("\n3. 🧪 Run system tests:")
    print("   python test_system.py")
    print("\n4. 📚 Add your own documents:")
    print("   - Upload files through the web interface")
    print("   - Process folders with documentation")
    print("   - Add web URLs or GitHub files")
    print("\n5. ⚙️ Configure LLM settings:")
    print("   - Choose your preferred provider")
    print("   - Adjust model parameters")
    print("   - Set temperature and token limits")


def main():
    """Main quick start function."""
    print("🚀 Documentation Copilot - Quick Start")
    print("=" * 50)
    
    # Check environment
    has_api_keys = check_environment()
    
    # Create sample data
    sample_files = create_sample_data()
    if not sample_files:
        print("❌ Failed to create sample data. Exiting.")
        return
    
    # Process documents
    chunks = process_documents(sample_files)
    if not chunks:
        print("❌ Failed to process documents. Exiting.")
        return
    
    # Set up vector store
    vector_store = setup_vector_store(chunks)
    if not vector_store:
        print("❌ Failed to set up vector store. Exiting.")
        return
    
    # Test RAG functionality
    rag_engine = test_rag_functionality(vector_store)
    
    # Show next steps
    show_next_steps()
    
    print("\n✨ You're all set! The Documentation Copilot is ready to use.")


if __name__ == "__main__":
    main() 