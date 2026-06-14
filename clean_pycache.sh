#!/bin/bash
# PPC8 - Clean all __pycache__ directories and .pyc files
# Usage: ./clean_pycache.sh

echo "Cleaning up __pycache__ directories and .pyc files..."

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

echo "Cleanup complete."
