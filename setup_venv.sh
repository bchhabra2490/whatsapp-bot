#!/bin/bash
# Setup virtual environment for the project

set -e

echo "🔧 Setting up virtual environment..."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating .venv directory..."
    python3 -m venv .venv
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the venv manually, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or on Windows:"
echo "  .venv\\Scripts\\activate"
