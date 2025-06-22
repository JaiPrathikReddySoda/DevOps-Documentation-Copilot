# DevOps & Documentation Copilot

A comprehensive AI-powered documentation assistant that processes various document types, stores them as embeddings in a FAISS vector database, and provides intelligent answers through a web interface or Slack bot using RAG.

This Documentation Copilot transforms your documentation into an intelligent knowledge base that can:

- **Process Multiple Document Types**: Markdown, PDF, Word docs, text files, web URLs, and GitHub files
- **Create Smart Embeddings**: Break documents into meaningful chunks and store them as vectors
- **Provide Intelligent Answers**: Use RAG to find relevant information and generate contextual responses
- **Show Source Attribution**: Always cite which documents were used to generate answers
- **Work Everywhere**: Access through a web interface or directly in Slack

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │  Document       │    │   Vector        │
│   (PDF, MD,     │───▶│  Processor      │───▶│   Store         │
│   URLs, etc.)   │    │  (Chunking)     │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   User Query    │───▶│   RAG Engine    │◀──────────┘
│   (Web/Slack)   │    │   (LLM + RAG)   │
└─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **API Keys** for at least one LLM provider (OpenAI, Groq, or Anthropic)
- **Slack Workspace** (optional, for Slack bot functionality)

## 🛠️ Installation & Setup

### Step 1: Clone and Navigate to Project

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd doc-copilot

# Or if you're already in the project directory
pwd  # Should show your project path
```

### Step 2: Create Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in your project root with your API keys:

```bash
# Create .env file
touch .env
```

Add the following content to your `.env` file:

```env
# Required: At least one LLM provider API key
OPENAI_API_KEY=your_openai_api_key_here
# OR
GROQ_API_KEY=your_groq_api_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Slack bot configuration (only if using Slack)
SLACK_BOT_TOKEN=your_slack_bot_token_here
SLACK_APP_TOKEN=your_slack_app_token_here

# Optional: Default configuration
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=1000
```

**How to Get API Keys:**

1. **OpenAI API Key**: 
   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Sign up/login and create a new API key

2. **Groq API Key**:
   - Visit [Groq Console](https://console.groq.com/)
   - Sign up and generate an API key

3. **Anthropic API Key**:
   - Go to [Anthropic Console](https://console.anthropic.com/)
   - Sign up and create an API key

### Step 5: Quick Setup (Optional)

Run the automated setup script to verify everything is working:

```bash
python quick_start.py
```

This script will:
- Validate your API keys
- Test document processing
- Create a sample vector store
- Verify the RAG system

## Running the System

### Option 1: Web Interface (Recommended for First Use)

Start the Streamlit web application:

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

**Using the Web Interface:**

1. **Upload Documents**: 
   - Click "Browse files" to upload individual files
   - Or enter a folder path to process all documents in that folder
   - Supported formats: PDF, Markdown, Word docs, text files

2. **Process Web Content**:
   - Enter a URL to scrape and process web content
   - Or enter a GitHub URL to process repository files

3. **Ask Questions**:
   - Type your question in the chat interface
   - Select your preferred LLM provider and model
   - Get answers with source citations

### Option 2: Slack Bot (For Team Collaboration)

#### Step 1: Create Slack App

1. Go to [Slack API Apps](https://api.slack.com/apps)
2. Click "Create New App" → "From scratch"
3. Name your app (e.g., "Documentation Copilot")
4. Select your workspace

#### Step 2: Configure Slack App

1. **Enable Socket Mode**:
   - Go to "Socket Mode" in the left sidebar
   - Enable Socket Mode
   - Generate an App-Level Token (starts with `xapp-`)

2. **Add Bot Token Scopes**:
   - Go to "OAuth & Permissions"
   - Add these Bot Token Scopes:
     - `commands` (for slash commands)
     - `chat:write` (to send messages)
     - `app_mentions:read` (to respond to mentions)

3. **Install App to Workspace**:
   - Click "Install to Workspace"
   - Copy the Bot User OAuth Token (starts with `xoxb-`)

4. **Create Slash Commands**:
   - Go to "Slash Commands"
   - Create these commands:
     - `/ask` - Ask questions about documents
     - `/docs-status` - Check system status
     - `/docs-help` - Show help information

#### Step 3: Update Environment Variables

Add your Slack tokens to your `.env` file:

```env
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
```

#### Step 4: Run the Slack Bot

```bash
python slack_bot.py
```

**Using the Slack Bot:**

1. **Invite the bot** to a channel: `@your-bot-name`
2. **Check status**: Type `/docs-status`
3. **Ask questions**: Type `/ask What is this project about?`
4. **Get help**: Type `/docs-help`

## 📚 Supported Document Types

| Format | Extension | Features |
|--------|-----------|----------|
| **Markdown** | `.md` | Direct parsing, metadata extraction |
| **PDF** | `.pdf` | Text extraction, layout preservation |
| **Word** | `.docx` | Full text and formatting |
| **Text** | `.txt` | Simple text processing |
| **Web URLs** | - | Content scraping, metadata |
| **GitHub Files** | - | Direct repository access |

## 🔧 Configuration Options

### LLM Provider Settings

You can configure different LLM providers in the web interface or modify defaults in your `.env` file:

```env
# Default LLM Configuration
DEFAULT_LLM_PROVIDER=openai  # openai, groq, anthropic
DEFAULT_MODEL=gpt-3.5-turbo  # Model name for the provider
DEFAULT_TEMPERATURE=0.1      # Creativity level (0.0-1.0)
DEFAULT_MAX_TOKENS=1000      # Maximum response length
```

### Vector Store Settings

The system automatically manages:
- **Chunk Size**: 1000 characters per chunk
- **Overlap**: 200 characters between chunks
- **Similarity Threshold**: 0.5 (configurable)
- **Top-K Results**: 5 (configurable)

## Testing the System

### Automated Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

This will test:
- Document processing
- Vector storage
- RAG functionality
- API integrations
- Error handling

### Manual Testing

1. **Test Document Processing**:
   ```bash
   python -c "
   from document_processor import DocumentProcessor
   processor = DocumentProcessor()
   result = processor.process_file('sample_docs/README.md')
   print(f'Processed {len(result)} chunks')
   "
   ```

2. **Test Vector Store**:
   ```bash
   python -c "
   from vector_store import VectorStore
   store = VectorStore()
   stats = store.get_stats()
   print(f'Total vectors: {stats[\"total_vectors\"]}')
   "
   ```

3. **Test RAG Engine**:
   ```bash
   python -c "
   from rag_engine import RAGEngine
   from vector_store import VectorStore
   store = VectorStore()
   store.load()
   rag = RAGEngine(store)
   result = rag.answer_question('What is this project about?')
   print(f'Answer: {result[\"answer\"]}')
   "
   ```

## Monitoring and Logs

### Log Files

The system creates logs in the `logs/` directory:
- `app.log` - Web application logs
- `slack_bot.log` - Slack bot logs
- `document_processor.log` - Document processing logs

### System Status

Check system status through:
- **Web Interface**: Status panel shows document count and system health
- **Slack Bot**: `/docs-status` command
- **Direct API**: Use the utility functions in `utils.py`

## Troubleshooting

### Common Issues

1. **"API Key Not Found" Error**:
   - Ensure your `.env` file exists and contains valid API keys
   - Check that the virtual environment is activated
   - Verify API key format (no extra spaces or quotes)

2. **"No Documents Loaded" Error**:
   - Upload documents through the web interface first
   - Check that documents are in supported formats
   - Verify file permissions

3. **Slack Bot Not Responding**:
   - Ensure bot is running: `python slack_bot.py`
   - Check Slack app configuration and permissions
   - Verify tokens in `.env` file
   - Check bot is invited to the channel

4. **Vector Store Corruption**:
   - Delete `vector_index/` directory to reset
   - Re-upload documents
   - Check available disk space

### Performance Optimization

- **Large Documents**: Break into smaller files for better processing
- **Many Documents**: Process in batches to avoid memory issues
- **Slow Responses**: Reduce `max_tokens` or use faster models
- **Memory Issues**: Increase system RAM or reduce chunk size

## Updating the System

To update dependencies:

```bash
# Update all packages
pip install -r requirements.txt --upgrade

# Or update specific packages
pip install --upgrade streamlit openai faiss-cpu
```

## 📁 Project Structure

```
doc-copilot/
├── app.py                 # Main Streamlit web application
├── slack_bot.py          # Slack bot implementation
├── document_processor.py # Document parsing and chunking
├── vector_store.py       # FAISS vector database operations
├── rag_engine.py         # RAG implementation with LLM integration
├── utils.py              # Utility functions and helpers
├── quick_start.py        # Automated setup and testing
├── test_system.py        # Comprehensive test suite
├── setup.py              # Installation and setup utilities
├── requirements.txt      # Python dependencies
├── slack_manifest.yaml   # Slack app configuration
├── .env                  # Environment variables (create this)
├── vector_index/         # Vector database storage
├── sample_docs/          # Example documents
├── logs/                 # Application logs
└── README.md            # This file
```

## Contributing

This is a hackathon MVP. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source. Feel free to use, modify, and distribute.

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs in the `logs/` directory
3. Run the test suite: `python test_system.py`
4. Check your API keys and environment configuration

---

**Happy Documenting!✨** 