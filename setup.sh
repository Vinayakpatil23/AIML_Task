#!/bin/bash

# Heart Disease Classification - Setup and Run Script
# This script sets up the environment and runs all components

echo "=========================================="
echo "Heart Disease Classification System"
echo "=========================================="
echo ""

# Check Python installation
echo "✓ Checking Python..."
python --version

# Install dependencies
echo ""
echo "✓ Installing dependencies..."
pip install -r requirements.txt -q

# Run preprocessing
echo ""
echo "✓ Step 1: Preprocessing data..."
python 1_preprocessing.py

# Run model training
echo ""
echo "✓ Step 2: Training models..."
python 2_train_model.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the system:"
echo "1. Start FastAPI server (in one terminal):"
echo "   python 3_fastapi_app.py"
echo ""
echo "2. Start Streamlit app (in another terminal):"
echo "   streamlit run 4_streamlit_app.py"
echo ""
echo "3. Access Streamlit at: http://localhost:8501"
echo "   FastAPI docs at: http://localhost:8000/docs"
echo ""
