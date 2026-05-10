#!/bin/bash
# Quick launcher for the Streamlit app.
# Assumes .venv exists and ollama is already running in another terminal.

set -e

if [ ! -d ".venv" ]; then
    echo "No .venv found. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate
streamlit run app.py
