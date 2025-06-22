"""
RAG (Retrieval-Augmented Generation) Engine

This module implements the core RAG functionality:
- Retrieving relevant document chunks based on user queries
- Generating answers using large language models
- Providing source attribution for answers
- Supporting multiple LLM providers (OpenAI, Groq, Anthropic)
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import json


class RAGEngine:
    """
    RAG Engine for document question answering.
    
    This class handles:
    - Retrieving relevant document chunks
    - Generating answers using LLMs
    - Managing different LLM providers
    - Providing source attribution
    """
    
    def __init__(self, 
                 vector_store,
                 llm_provider: str = "openai",
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        """
        Initialize the RAG engine.
        
        Args:
            vector_store: VectorStore instance for document retrieval
            llm_provider: LLM provider ("openai", "groq", "anthropic")
            model_name: Model name for the selected provider
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Define the system prompt for RAG
        self.system_prompt = """You are a helpful documentation assistant. Your task is to answer questions based on the provided document context.

IMPORTANT GUIDELINES:
1. Only answer based on the information provided in the context
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided documents."
3. Always cite the specific sources you used from the context
4. Be concise but comprehensive
5. If you're unsure about something, acknowledge the uncertainty
6. Use markdown formatting for better readability

Context from documents:
{context}

Question: {question}

Please provide a helpful answer based on the context above:"""
    
    def _initialize_llm(self):
        """Initialize the LLM based on the selected provider."""
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key
            )
        
        elif self.llm_provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key
            )
        
        elif self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def answer_question(self, 
                       question: str, 
                       k: int = 5, 
                       threshold: float = 0.5,
                       include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
            threshold: Similarity threshold for retrieval
            include_sources: Whether to include source information
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Retrieve relevant document chunks
        relevant_chunks = self.vector_store.search(question, k, threshold)
        
        if not relevant_chunks:
            return {
                'answer': "I don't have enough information to answer this question. No relevant documents were found in the knowledge base.",
                'sources': [],
                'chunks_used': 0,
                'confidence': 0.0
            }
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            source_info = self._format_source_info(chunk['metadata'])
            context_parts.append(f"Document {i+1} ({source_info}):\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Generate answer using LLM
        try:
            messages = [
                SystemMessage(content=self.system_prompt.format(context=context, question=question)),
                HumanMessage(content=question)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Calculate confidence based on average similarity score
            avg_score = sum(chunk['score'] for chunk in relevant_chunks) / len(relevant_chunks)
            
            # Prepare sources information
            sources = []
            if include_sources:
                sources = [self._format_source_info(chunk['metadata']) for chunk in relevant_chunks]
            
            return {
                'answer': answer,
                'sources': sources,
                'chunks_used': len(relevant_chunks),
                'confidence': avg_score,
                'raw_chunks': relevant_chunks if include_sources else []
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'chunks_used': 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def answer_question_with_filters(self, 
                                   question: str,
                                   k: int = 5,
                                   threshold: float = 0.5,
                                   file_types: Optional[List[str]] = None,
                                   sources: Optional[List[str]] = None,
                                   include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a question with additional filters.
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
            threshold: Similarity threshold for retrieval
            file_types: Filter by file types
            sources: Filter by source files
            include_sources: Whether to include source information
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Retrieve relevant document chunks with filters
        relevant_chunks = self.vector_store.search_with_filters(
            question, k, threshold, file_types, sources
        )
        
        if not relevant_chunks:
            filter_info = []
            if file_types:
                filter_info.append(f"file types: {', '.join(file_types)}")
            if sources:
                filter_info.append(f"sources: {', '.join(sources)}")
            
            filter_msg = f" with filters: {', '.join(filter_info)}" if filter_info else ""
            
            return {
                'answer': f"I don't have enough information to answer this question{filter_msg}. No relevant documents were found in the knowledge base.",
                'sources': [],
                'chunks_used': 0,
                'confidence': 0.0
            }
        
        # Use the same logic as answer_question
        return self.answer_question(question, k, threshold, include_sources)
    
    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        """
        Format source information for display.
        
        Args:
            metadata: Chunk metadata
            
        Returns:
            Formatted source string
        """
        source_type = metadata.get('type', 'unknown')
        source = metadata.get('source', 'unknown')
        
        if source_type == 'markdown':
            filename = metadata.get('filename', '')
            return f"Markdown: {filename}"
        
        elif source_type == 'pdf':
            filename = metadata.get('filename', '')
            page = metadata.get('page', '')
            return f"PDF: {filename} (Page {page})"
        
        elif source_type == 'docx':
            filename = metadata.get('filename', '')
            return f"Word: {filename}"
        
        elif source_type == 'text':
            filename = metadata.get('filename', '')
            return f"Text: {filename}"
        
        elif source_type == 'web_url':
            title = metadata.get('title', '')
            return f"Web: {title}"
        
        elif source_type == 'github_file':
            file_path = metadata.get('file_path', '')
            return f"GitHub: {file_path}"
        
        else:
            return f"{source_type}: {source}"
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get available models for each provider.
        
        Returns:
            Dictionary mapping providers to available models
        """
        return {
            "openai": [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4-32k"
            ],
            "groq": [
                "llama2-70b-4096",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ],
            "anthropic": [
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307"
            ]
        }
    
    def update_llm_config(self, 
                         llm_provider: str = None,
                         model_name: str = None,
                         temperature: float = None,
                         max_tokens: int = None) -> None:
        """
        Update LLM configuration.
        
        Args:
            llm_provider: New LLM provider
            model_name: New model name
            temperature: New temperature
            max_tokens: New max tokens
        """
        if llm_provider is not None:
            self.llm_provider = llm_provider
        
        if model_name is not None:
            self.model_name = model_name
        
        if temperature is not None:
            self.temperature = temperature
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
        
        # Reinitialize LLM with new configuration
        self.llm = self._initialize_llm()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current RAG engine configuration.
        
        Returns:
            Dictionary with current configuration
        """
        return {
            'llm_provider': self.llm_provider,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'vector_store_stats': self.vector_store.get_stats()
        }
    
    def batch_answer_questions(self, 
                             questions: List[str],
                             k: int = 5,
                             threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions to answer
            k: Number of relevant chunks to retrieve per question
            threshold: Similarity threshold for retrieval
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for question in questions:
            result = self.answer_question(question, k, threshold)
            result['question'] = question
            results.append(result)
        
        return results 