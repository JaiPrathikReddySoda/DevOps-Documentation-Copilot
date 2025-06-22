"""
Setup Script for Documentation Copilot

This script sets up the Documentation Copilot MVP:
1. Installs required dependencies
2. Creates necessary directories
3. Sets up environment configuration
4. Runs initial tests
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("🔧 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {str(e)}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "vector_index",
        "sample_docs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_env_file():
    """Create .env file if it doesn't exist."""
    print("\n⚙️ Setting up environment configuration...")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create .env file from template
    env_template = """# Documentation Copilot Environment Variables
# Fill in your API keys below

# LLM API Keys (set at least one)
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Slack Bot Configuration (optional)
SLACK_BOT_TOKEN=your_slack_bot_token_here
SLACK_APP_TOKEN=your_slack_app_token_here

# Default LLM Configuration
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=1000

# Vector Store Configuration
VECTOR_STORE_PATH=vector_index
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Document Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_template)
        print("✅ Created .env file")
        print("   Please edit .env and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {str(e)}")
        return False


def run_tests():
    """Run system tests."""
    print("\n🧪 Running system tests...")
    
    try:
        # Import test modules to check if everything is working
        from test_system import main as run_tests
        run_tests()
        print("✅ All tests passed")
        return True
    except Exception as e:
        print(f"⚠️  Some tests failed: {str(e)}")
        print("   This is normal if API keys are not set")
        return False


def show_next_steps():
    """Show next steps for the user."""
    print("\n🎉 Setup completed!")
    print("\n📋 Next Steps:")
    print("1. 🔑 Set up API keys:")
    print("   - Edit the .env file")
    print("   - Add your OpenAI, Groq, or Anthropic API key")
    print("\n2. 🚀 Start the web interface:")
    print("   streamlit run app.py")
    print("\n3. 🤖 Start the Slack bot (optional):")
    print("   python slack_bot.py")
    print("\n4. 🧪 Run quick start:")
    print("   python quick_start.py")
    print("\n5. 📚 Add your documents:")
    print("   - Upload files through the web interface")
    print("   - Process folders with documentation")
    print("   - Add web URLs or GitHub files")


def main():
    """Main setup function."""
    print("🚀 Documentation Copilot - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create environment file
    if not create_env_file():
        print("❌ Failed to create environment file.")
        sys.exit(1)
    
    # Run tests
    run_tests()
    
    # Show next steps
    show_next_steps()
    
    print("\n✨ Setup completed successfully!")
    print("   You can now start using the Documentation Copilot.")


if __name__ == "__main__":
    main() 