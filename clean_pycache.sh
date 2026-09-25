#!/bin/bash
# PPC8 - Clean all __pycache__ directories, .pyc/.pyo files, and tool caches
# Usage: ./clean_pycache.sh

set -e

echo "Cleaning up __pycache__ directories and .pyc/.pyo files..."

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

echo ""
echo "Cleaning up ruff cache (.ruff_cache)..."

find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +

echo ""
echo "Cleaning up mypy cache (.mypy_cache)..."

find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

echo ""
echo "Cleanup complete."
