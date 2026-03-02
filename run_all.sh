#!/bin/bash
# run_all.sh  —  Convenience script to run the full Q1 pipeline
# Usage: bash run_all.sh [fast|core]

MODE=${1:-full}

echo "========================================================"
echo "  Lightweight Multi-Agent Misinformation Detection"
echo "  Q1 Pipeline Runner"
echo "========================================================"
echo ""

# Check dataset exists
if [ ! -f "data/raw/train.tsv" ]; then
    echo "ERROR: data/raw/train.tsv not found."
    echo ""
    echo "Please download the LIAR dataset:"
    echo "  wget https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    echo "  unzip liar_dataset.zip"
    echo "  mv *.tsv data/raw/"
    exit 1
fi

echo "Dataset found. Starting pipeline in mode: $MODE"
echo ""

if [ "$MODE" = "fast" ]; then
    python main.py --fast
elif [ "$MODE" = "core" ]; then
    python main.py --core-only
else
    python main.py
fi

echo ""
echo "Done. Results saved in results/"
echo "  Q1 Summary : results/q1_summary.txt"
echo "  LaTeX Table: results/latex_table.tex"
