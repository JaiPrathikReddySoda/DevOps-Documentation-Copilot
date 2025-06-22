"""
Documentation Copilot - Streamlit Web Application

This is the main web interface for the Documentation Copilot MVP.
It provides a user-friendly way to:
- Upload and process documents
- Ask questions about the documents
- Get AI-powered answers with source attribution
- Configure LLM settings
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
from utils import (
    validate_file_path, validate_folder_path, validate_url, validate_github_url,
    get_supported_file_extensions, is_supported_file, create_sample_documents,
    validate_api_keys, get_environment_info, format_file_size
)

# Page configuration
st.set_page_config(
    page_title="Documentation Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .source-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 3px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def initialize_components():
    """Initialize vector store and RAG engine components."""
    try:
        # Initialize vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
            
            # Try to load existing index
            if st.session_state.vector_store.load():
                st.session_state.documents_loaded = True
                st.success("Loaded existing document index!")
        
        # Initialize RAG engine
        if st.session_state.rag_engine is None:
            st.session_state.rag_engine = RAGEngine(
                vector_store=st.session_state.vector_store,
                llm_provider=st.session_state.get('llm_provider', 'openai'),
                model_name=st.session_state.get('model_name', 'gpt-3.5-turbo'),
                temperature=st.session_state.get('temperature', 0.1),
                max_tokens=st.session_state.get('max_tokens', 1000)
            )
    
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")


def process_uploaded_files(uploaded_files: List) -> List[Dict[str, Any]]:
    """Process uploaded files and return chunks."""
    processor = DocumentProcessor()
    all_chunks = []
    
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process the file
                if is_supported_file(tmp_path):
                    chunks = processor.process_file(tmp_path)
                    all_chunks.extend(chunks)
                    st.success(f"Processed {uploaded_file.name}: {len(chunks)} chunks")
                else:
                    st.warning(f"Skipped {uploaded_file.name}: Unsupported file type")
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return all_chunks


def process_folder_path(folder_path: str) -> List[Dict[str, Any]]:
    """Process all files in a folder."""
    processor = DocumentProcessor()
    
    with st.spinner(f"Processing folder: {folder_path}"):
        try:
            chunks = processor.process_folder(folder_path)
            st.success(f"Processed folder: {len(chunks)} chunks from {folder_path}")
            return chunks
        except Exception as e:
            st.error(f"Error processing folder: {str(e)}")
            return []


def process_url(url: str) -> List[Dict[str, Any]]:
    """Process content from a URL."""
    processor = DocumentProcessor()
    
    with st.spinner(f"Processing URL: {url}"):
        try:
            chunks = processor.process_url(url)
            st.success(f"Processed URL: {len(chunks)} chunks from {url}")
            return chunks
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")
            return []


def process_github_file(repo_url: str, file_path: str) -> List[Dict[str, Any]]:
    """Process a file from GitHub."""
    processor = DocumentProcessor()
    
    with st.spinner(f"Processing GitHub file: {repo_url}/{file_path}"):
        try:
            chunks = processor.process_github_file(repo_url, file_path)
            st.success(f"Processed GitHub file: {len(chunks)} chunks")
            return chunks
        except Exception as e:
            st.error(f"Error processing GitHub file: {str(e)}")
            return []


def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">📚 Documentation Copilot</h1>', unsafe_allow_html=True)
    
    # Initialize components
    initialize_components()
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h3 class="sub-header">⚙️ Configuration</h3>', unsafe_allow_html=True)
        
        # LLM Configuration
        st.subheader("LLM Settings")
        
        # Check API keys
        api_keys = validate_api_keys()
        if not any(api_keys.values()):
            st.error("No API keys found! Please set at least one API key.")
            st.info("Set environment variables: OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY")
        else:
            # LLM Provider selection
            available_providers = [k for k, v in api_keys.items() if v]
            llm_provider = st.selectbox(
                "LLM Provider",
                available_providers,
                index=available_providers.index(st.session_state.get('llm_provider', 'openai'))
            )
            
            # Model selection
            if st.session_state.rag_engine:
                available_models = st.session_state.rag_engine.get_available_models()
                models = available_models.get(llm_provider, [])
                if models:
                    # Check if the session state model is valid for the current provider
                    current_model_in_session = st.session_state.get('model_name')
                    model_index = 0
                    if current_model_in_session in models:
                        model_index = models.index(current_model_in_session)

                    model_name = st.selectbox(
                        "Model",
                        models,
                        index=model_index
                    )
                else:
                    model_name = st.text_input("Model Name", value=st.session_state.get('model_name', ''))
            else:
                model_name = st.text_input("Model Name", value=st.session_state.get('model_name', ''))
            
            # Other parameters
            temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.get('temperature', 0.1), 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4000, st.session_state.get('max_tokens', 1000), 100)
            
            # Update configuration if changed
            if st.button("Update LLM Config"):
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.update_llm_config(
                        llm_provider=llm_provider,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    st.session_state.llm_provider = llm_provider
                    st.session_state.model_name = model_name
                    st.session_state.temperature = temperature
                    st.session_state.max_tokens = max_tokens
                    st.success("LLM configuration updated!")
        
        # Vector Store Stats
        if st.session_state.vector_store:
            st.subheader("📊 Vector Store Stats")
            stats = st.session_state.vector_store.get_stats()
            
            st.metric("Total Vectors", stats['total_vectors'])
            st.metric("Unique Sources", stats['unique_sources'])
            st.metric("Total Tokens", f"{stats['total_tokens']:,}")
            
            if stats['file_types']:
                st.write("**File Types:**")
                for file_type, count in stats['file_types'].items():
                    st.write(f"• {file_type}: {count}")
        
        # Actions
        st.subheader("🛠️ Actions")
        
        if st.button("Clear All Data"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear()
                st.session_state.documents_loaded = False
                st.session_state.chat_history = []
                st.success("All data cleared!")
        
        if st.button("Save Vector Store"):
            if st.session_state.vector_store:
                st.session_state.vector_store.save()
                st.success("Vector store saved!")
        
        if st.button("Create Sample Documents"):
            sample_files = create_sample_documents()
            st.success(f"Created {len(sample_files)} sample documents!")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Documents", "💬 Ask Questions", "📋 Chat History", "ℹ️ About"])
    
    # Tab 1: Upload Documents
    with tab1:
        st.markdown('<h2 class="sub-header">Upload Documents</h2>', unsafe_allow_html=True)
        
        # File upload
        st.subheader("📤 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=get_supported_file_extensions(),
            accept_multiple_files=True,
            help="Supported formats: Markdown (.md), PDF (.pdf), Word (.docx), Text (.txt)"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                chunks = process_uploaded_files(uploaded_files)
                if chunks and st.session_state.vector_store:
                    st.session_state.vector_store.add_documents(chunks)
                    st.session_state.documents_loaded = True
                    st.session_state.vector_store.save()
                    st.success(f"Successfully processed {len(chunks)} chunks!")
        
        # Folder path input
        st.subheader("📂 Process Folder")
        folder_path = st.text_input(
            "Enter folder path",
            placeholder="/path/to/your/documents",
            help="Enter the full path to a folder containing documents"
        )
        
        if folder_path:
            if validate_folder_path(folder_path):
                if st.button("Process Folder"):
                    chunks = process_folder_path(folder_path)
                    if chunks and st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(chunks)
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store.save()
                        st.success(f"Successfully processed {len(chunks)} chunks!")
            else:
                st.error("Invalid folder path")
        
        # URL input
        st.subheader("🌐 Process Web URL")
        url = st.text_input(
            "Enter URL",
            placeholder="https://example.com/documentation",
            help="Enter a web URL to scrape and process"
        )
        
        if url:
            if validate_url(url):
                if st.button("Process URL"):
                    chunks = process_url(url)
                    if chunks and st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(chunks)
                        st.session_state.documents_loaded = True
                        st.session_state.vector_store.save()
                        st.success(f"Successfully processed {len(chunks)} chunks!")
            else:
                st.error("Invalid URL")
        
        # GitHub file input
        st.subheader("🐙 Process GitHub File")
        github_url = st.text_input(
            "GitHub File URL",
            placeholder="https://github.com/user/repo/blob/main/README.md",
            help="Enter the full URL to a file on GitHub"
        )
        
        if github_url:
            if validate_github_url(github_url):
                if st.button("Process GitHub File"):
                    try:
                        processor = DocumentProcessor()
                        chunks = processor.process_github_file(github_url)
                        if chunks and st.session_state.vector_store:
                            st.session_state.vector_store.add_documents(chunks)
                            st.session_state.documents_loaded = True
                            st.session_state.vector_store.save()
                            st.success(f"Successfully processed {len(chunks)} chunks!")
                    except Exception as e:
                        st.error(f"Error processing GitHub file: {str(e)}")
            else:
                st.error("Invalid GitHub URL")
    
    # Tab 2: Ask Questions
    with tab2:
        st.markdown('<h2 class="sub-header">Ask Questions</h2>', unsafe_allow_html=True)
        
        if not st.session_state.documents_loaded:
            st.warning("No documents loaded. Please upload some documents first.")
        else:
            # Question input
            question = st.text_area(
                "Ask a question about your documents",
                placeholder="What is the main feature of this system?",
                height=100
            )
            
            # Search parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                k_results = st.number_input("Number of results (k)", 1, 20, 5, 1)
            
            with col2:
                threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.05, help="Minimum similarity score for a document chunk to be considered relevant. Lower values retrieve more documents, but they may be less relevant.")
            
            with col3:
                include_sources = st.checkbox("Include sources", value=True)
            
            if st.button("Ask Question", type="primary"):
                if question.strip():
                    with st.spinner("Generating answer..."):
                        try:
                            result = st.session_state.rag_engine.answer_question(
                                question=question,
                                k=k_results,
                                threshold=threshold,
                                include_sources=include_sources
                            )
                            
                            # Display answer
                            st.markdown("### Answer")
                            st.markdown(result['answer'])
                            
                            # Display metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Chunks Used", result['chunks_used'])
                            with col2:
                                st.metric("Confidence", f"{result['confidence']:.2f}")
                            with col3:
                                st.metric("Sources", len(result['sources']))
                            
                            # Display sources
                            if result['sources']:
                                st.markdown("### Sources")
                                for source in result['sources']:
                                    st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'question': question,
                                'answer': result['answer'],
                                'sources': result['sources'],
                                'confidence': result['confidence'],
                                'timestamp': time.time()
                            })
                            
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                else:
                    st.warning("Please enter a question.")
    
    # Tab 3: Chat History
    with tab3:
        st.markdown('<h2 class="sub-header">Chat History</h2>', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.info("No chat history yet. Ask some questions to see them here!")
        else:
            # Clear chat history button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Display chat history
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:50]}...", expanded=False):
                    st.markdown("**Question:**")
                    st.write(chat['question'])
                    
                    st.markdown("**Answer:**")
                    st.markdown(chat['answer'])
                    
                    if chat['sources']:
                        st.markdown("**Sources:**")
                        for source in chat['sources']:
                            st.markdown(f'<div class="source-item">{source}</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{chat['confidence']:.2f}")
                    with col2:
                        st.metric("Sources", len(chat['sources']))
    
    # Tab 4: About
    with tab4:
        st.markdown('<h2 class="sub-header">About Documentation Copilot</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        **Documentation Copilot** is an AI-powered tool that helps you understand and navigate your documentation.
        
        ### Features
        - 📄 **Multi-format Support**: Markdown, PDF, Word docs, text files, web URLs, and GitHub files
        - 🔍 **Smart Search**: Vector-based similarity search for relevant content
        - 🤖 **AI Answers**: Powered by OpenAI, Groq, or Anthropic LLMs
        - 📚 **Source Attribution**: Always know where answers come from
        - 💾 **Local Storage**: FAISS vector database for fast retrieval
        
        ### How it Works
        1. **Upload Documents**: Add your documentation files
        2. **Processing**: Documents are chunked and embedded into vectors
        3. **Ask Questions**: Query your documentation in natural language
        4. **Get Answers**: Receive AI-generated answers with source references
        
        ### Supported File Types
        - Markdown (.md)
        - PDF (.pdf)
        - Word Documents (.docx)
        - Text Files (.txt)
        - Web URLs
        - GitHub Files
        """)
        
        # Environment information
        st.subheader("Environment Information")
        env_info = get_environment_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Python Version:** {env_info['python_version']}")
            st.write(f"**Platform:** {env_info['platform']}")
        
        with col2:
            st.write(f"**Architecture:** {env_info['architecture']}")
            st.write(f"**Working Directory:** {env_info['working_directory']}")
        
        # API Key status
        st.subheader("API Key Status")
        api_status = validate_api_keys()
        for provider, status in api_status.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {provider.title()}: {'Available' if status else 'Not Set'}")


if __name__ == "__main__":
    main() 