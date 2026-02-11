#!/bin/bash

set -e  # Stop on first error

echo "======================================="
echo "  End-to-End ASR Pipeline Execution"
echo "======================================="

# Move to project root safely
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

# --------------------------------------------------
# Step 1: Create virtual environment
# --------------------------------------------------
if [ ! -d "venv" ]; then
    echo "[1/6] Creating virtual environment..."
    python -m venv venv
else
    echo "[1/6] Virtual environment already exists."
fi

# --------------------------------------------------
# Step 2: Activate virtual environment
# --------------------------------------------------
echo "[2/6] Activating virtual environment..."
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# --------------------------------------------------
# Step 3: Upgrade pip
# --------------------------------------------------
echo "[3/6] Upgrading pip..."
python -m pip install --upgrade pip

# --------------------------------------------------
# Step 4: Install requirements
# --------------------------------------------------
echo "[4/6] Installing dependencies..."
pip install -r requirements.txt

# --------------------------------------------------
# Step 5: Run ASR training notebook (headless)
# --------------------------------------------------
echo "[5/6] Running ASR training notebook..."

jupyter nbconvert \
    --to notebook \
    --execute notebooks/05_asr_training.ipynb \
    --output executed_05_asr_training.ipynb

# --------------------------------------------------
# Step 6: Finish
# --------------------------------------------------
echo "[6/6] Pipeline completed successfully."
echo "Check executed notebook for outputs."

echo "======================================="
echo "  ASR Pipeline Finished"
echo "======================================="
