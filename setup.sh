#!/bin/bash

# Setup script for Cython Performance Suite
# Run this to install dependencies and build the project

set -e  # Exit on error

echo "🚀 Setting up Cython Performance Suite..."
echo "=========================================="

# Check Python version
echo "✓ Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Build Cython extensions
echo ""
echo "🔨 Building Cython extensions..."
python3 setup.py build_ext --inplace

# Check if build was successful
if [ -f "algorithms_cy.c" ] && [ -f "example_cy.c" ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Generated files:"
    ls -lh *.c *.so 2>/dev/null || ls -lh *.c *.pyd 2>/dev/null || echo "  (compiled extensions)"
    
    echo ""
    echo "📊 Annotation files (open in browser to see optimization details):"
    ls -lh *.html 2>/dev/null || echo "  (no annotation files - add annotate=True to setup.py)"
    
    echo ""
    echo "🎯 Ready to run!"
    echo ""
    echo "Try these commands:"
    echo "  python3 testing.py          # Original simple test"
    echo "  python3 benchmark.py        # Comprehensive benchmark suite"
    echo "  python3 profile_example.py  # Memory profiling"
    echo ""
else
    echo ""
    echo "❌ Build failed. Check error messages above."
    exit 1
fi
