#!/usr/bin/env python3
"""
Setup script for Pharmacy Problem Analyzer

This script helps users set up the application by checking dependencies
and guiding through configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment configuration."""
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not example_file.exists():
        print("❌ .env.example file not found")
        return False
    
    # Copy example to .env
    with open(example_file) as f:
        content = f.read()
    
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Created .env file from template")
    print("⚠️  Please edit .env file and add your API keys and MongoDB connection string")
    return True

def validate_config():
    """Validate configuration."""
    try:
        from config import Config
        Config.validate_config()
        print("✅ Configuration is valid")
        return True
    except ImportError:
        print("❌ Cannot import config module")
        return False
    except ValueError as e:
        print(f"⚠️  Configuration incomplete: {e}")
        print("   Please edit your .env file with the required values")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "temp_files",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Created necessary directories")

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your credentials:")
    print("   - GOOGLE_API_KEY: Your Google Gemini API key")
    print("   - MONGODB_URI: Your MongoDB Atlas connection string")
    print("\n2. Set up MongoDB Atlas Vector Search index:")
    print("   - Create index named 'Indexx' on 'embedding' field")
    print("   - Use 768 dimensions with cosine similarity")
    print("\n3. Start the application:")
    print("   - Web interface: streamlit run ragtrial.py")
    print("   - Upload API: python APIs/Uploadfilesapi.py")
    print("   - Query API: python APIs/QueryAPI.py")
    print("\n4. Test with example:")
    print("   - python example_usage.py")

def main():
    """Main setup process."""
    print("🏥 Pharmacy Problem Analyzer - Setup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Create directories
    create_directories()
    
    # Validate configuration (optional)
    validate_config()
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)