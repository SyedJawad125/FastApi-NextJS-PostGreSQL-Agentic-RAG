# ============================================================================
# setup.sh - Unix/Linux/Mac Setup Script
# ============================================================================
#!/bin/bash

echo "======================================"
echo "Advanced Agentic RAG System - Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/vectors
mkdir -p data/graphs
mkdir -p data/documents
mkdir -p data/cache
mkdir -p logs
mkdir -p uploads
mkdir -p uploaded_images

# Copy .env.example to .env if not exists
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your API keys"
fi

# Initialize database (if using)
echo ""
echo "Initializing database..."
python -c "from app.database import init_db; init_db()"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your GROQ_API_KEY"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python main.py"
echo "4. Visit: http://localhost:8000/docs"
echo ""