"""
Slack Bot for Documentation Copilot

This module implements a Slack bot that allows users to:
- Ask questions about documents using slash commands
- Get AI-powered answers with source attribution
- Interact with the documentation copilot directly from Slack
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt.context.say import Say
from slack_bolt.context.ack import Ack
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
from utils import validate_api_keys, get_environment_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Slack app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Global variables for components
vector_store = None
rag_engine = None


def initialize_components():
    """Initialize vector store and RAG engine components."""
    global vector_store, rag_engine
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        
        # Try to load existing index
        if vector_store.load():
            logger.info("Loaded existing document index")
        else:
            logger.info("No existing index found, starting fresh")
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            vector_store=vector_store,
            llm_provider=os.environ.get("DEFAULT_LLM_PROVIDER", "openai"),
            model_name=os.environ.get("DEFAULT_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.environ.get("DEFAULT_TEMPERATURE", "0.1")),
            max_tokens=int(os.environ.get("DEFAULT_MAX_TOKENS", "1000"))
        )
        
        logger.info("Components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise


@app.command("/ask")
def handle_ask_command(ack: Ack, command: Dict[str, Any], say: Say):
    """
    Handle the /ask slash command.
    
    Args:
        ack: Slack acknowledgment function
        command: Command data from Slack
        say: Function to send message to channel
    """
    ack()
    
    # Extract question from command
    question = command.get("text", "").strip()
    
    if not question:
        say("Please provide a question. Usage: `/ask your question here`")
        return
    
    # Check if components are initialized
    if not rag_engine or not vector_store:
        say("❌ Documentation Copilot is not properly initialized. Please check the server logs.")
        return
    
    # Check if documents are loaded
    stats = vector_store.get_stats()
    if stats['total_vectors'] == 0:
        say("❌ No documents have been loaded yet. Please upload some documents first.")
        return
    
    try:
        # Generate answer
        result = rag_engine.answer_question(
            question=question,
            k=5,
            threshold=0.5,
            include_sources=True
        )
        
        # Format response
        response = format_slack_response(question, result)
        
        # Send response
        say(response)
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        say(f"❌ Error generating answer: {str(e)}")


@app.command("/docs-status")
def handle_status_command(ack: Ack, command: Dict[str, Any], say: Say):
    """
    Handle the /docs-status slash command to show system status.
    
    Args:
        ack: Slack acknowledgment function
        command: Command data from Slack
        say: Function to send message to channel
    """
    ack()
    
    if not vector_store:
        say("❌ Documentation Copilot is not properly initialized.")
        return
    
    try:
        stats = vector_store.get_stats()
        
        # Format status message
        status_msg = f"""
📊 **Documentation Copilot Status**

📚 **Documents Loaded:**
• Total Vectors: {stats['total_vectors']:,}
• Unique Sources: {stats['unique_sources']}
• Total Tokens: {stats['total_tokens']:,}

📄 **File Types:**
"""
        
        for file_type, count in stats['file_types'].items():
            status_msg += f"• {file_type}: {count}\n"
        
        status_msg += f"""
🤖 **LLM Configuration:**
• Provider: {rag_engine.llm_provider if rag_engine else 'Not set'}
• Model: {rag_engine.model_name if rag_engine else 'Not set'}

🔧 **System Info:**
• Vector Store: {'✅ Loaded' if vector_store else '❌ Not loaded'}
• RAG Engine: {'✅ Ready' if rag_engine else '❌ Not ready'}
"""
        
        say(status_msg)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        say(f"❌ Error getting status: {str(e)}")


@app.command("/docs-help")
def handle_help_command(ack: Ack, command: Dict[str, Any], say: Say):
    """
    Handle the /docs-help slash command to show help information.
    
    Args:
        ack: Slack acknowledgment function
        command: Command data from Slack
        say: Function to send message to channel
    """
    ack()
    
    help_msg = """
📚 **Documentation Copilot - Slack Bot Help**

**Available Commands:**

🔍 `/ask <question>` - Ask a question about your documents
   Example: `/ask What is the main feature of this system?`

📊 `/docs-status` - Show system status and document statistics

❓ `/docs-help` - Show this help message

**Features:**
• AI-powered answers based on your documentation
• Source attribution for all answers
• Support for multiple document formats
• Fast vector-based search

**Supported Document Types:**
• Markdown (.md)
• PDF (.pdf)
• Word Documents (.docx)
• Text Files (.txt)
• Web URLs
• GitHub Files

**Note:** Documents must be uploaded through the web interface before using the Slack bot.
"""
    
    say(help_msg)


def format_slack_response(question: str, result: Dict[str, Any]) -> str:
    """
    Format the RAG result for Slack display.
    
    Args:
        question: Original question
        result: RAG result dictionary
        
    Returns:
        Formatted Slack message
    """
    # Start with the answer
    response = f"🤖 **Answer:**\n{result['answer']}\n\n"
    
    # Add metadata
    response += f"📊 **Metadata:**\n"
    response += f"• Chunks Used: {result['chunks_used']}\n"
    response += f"• Confidence: {result['confidence']:.2f}\n"
    response += f"• Sources: {len(result['sources'])}\n\n"
    
    # Add sources if available
    if result['sources']:
        response += "📚 **Sources:**\n"
        for i, source in enumerate(result['sources'], 1):
            response += f"{i}. {source}\n"
    
    return response


@app.event("app_mention")
def handle_app_mention(event: Dict[str, Any], say: Say):
    """
    Handle when the bot is mentioned in a channel.
    
    Args:
        event: Event data from Slack
        say: Function to send message to channel
    """
    # Extract question from mention
    text = event.get("text", "")
    
    # Remove bot mention
    bot_user_id = event.get("bot_id") or app.client.auth_test()["user_id"]
    question = text.replace(f"<@{bot_user_id}>", "").strip()
    
    if not question:
        say("Hi! I'm the Documentation Copilot. Use `/ask your question` to ask me about your documents, or `/docs-help` for more information.")
        return
    
    # Use the same logic as /ask command
    if not rag_engine or not vector_store:
        say("❌ Documentation Copilot is not properly initialized. Please check the server logs.")
        return
    
    stats = vector_store.get_stats()
    if stats['total_vectors'] == 0:
        say("❌ No documents have been loaded yet. Please upload some documents first.")
        return
    
    try:
        result = rag_engine.answer_question(
            question=question,
            k=5,
            threshold=0.5,
            include_sources=True
        )
        
        response = format_slack_response(question, result)
        say(response)
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        say(f"❌ Error generating answer: {str(e)}")


@app.event("message")
def handle_message(event: Dict[str, Any], say: Say):
    """
    Handle direct messages to the bot.
    
    Args:
        event: Event data from Slack
        say: Function to send message to channel
    """
    # Only respond to direct messages
    if event.get("channel_type") != "im":
        return
    
    text = event.get("text", "").strip()
    
    if not text:
        return
    
    # Check if it looks like a question
    if text.endswith("?") or any(word in text.lower() for word in ["what", "how", "why", "when", "where", "who", "which"]):
        # Use the same logic as /ask command
        if not rag_engine or not vector_store:
            say("❌ Documentation Copilot is not properly initialized. Please check the server logs.")
            return
        
        stats = vector_store.get_stats()
        if stats['total_vectors'] == 0:
            say("❌ No documents have been loaded yet. Please upload some documents first.")
            return
        
        try:
            result = rag_engine.answer_question(
                question=text,
                k=5,
                threshold=0.5,
                include_sources=True
            )
            
            response = format_slack_response(text, result)
            say(response)
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            say(f"❌ Error generating answer: {str(e)}")
    else:
        say("Hi! I'm the Documentation Copilot. You can ask me questions about your documents, or use `/docs-help` for more information.")


def main():
    """Main function to run the Slack bot."""
    # Check required environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set SLACK_BOT_TOKEN and SLACK_APP_TOKEN")
        return
    
    # Check API keys
    api_keys = validate_api_keys()
    if not any(api_keys.values()):
        logger.error("No LLM API keys found! Please set at least one API key.")
        logger.error("Set environment variables: OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY")
        return
    
    # Initialize components
    try:
        initialize_components()
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        return
    
    # Get app token
    app_token = os.environ.get("SLACK_APP_TOKEN")
    
    # Start the bot
    logger.info("Starting Documentation Copilot Slack Bot...")
    
    try:
        handler = SocketModeHandler(app, app_token)
        handler.start()
    except Exception as e:
        logger.error(f"Error starting Slack bot: {str(e)}")


if __name__ == "__main__":
    main() 