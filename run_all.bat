@echo off
REM run_all.bat  —  Windows convenience script to run the full Q1 pipeline
REM Usage: run_all.bat [fast|core]

echo ========================================================
echo   Lightweight Multi-Agent Misinformation Detection
echo   Q1 Pipeline Runner
echo ========================================================
echo.

IF NOT EXIST "data\raw\train.tsv" (
    echo ERROR: data\raw\train.tsv not found.
    echo.
    echo Please download the LIAR dataset:
    echo   https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
    echo   Unzip and place train.tsv, valid.tsv, test.tsv in data\raw\
    pause
    exit /b 1
)

SET MODE=%1
IF "%MODE%"=="" SET MODE=full

echo Dataset found. Starting pipeline in mode: %MODE%
echo.

IF "%MODE%"=="fast" (
    python main.py --fast
) ELSE IF "%MODE%"=="core" (
    python main.py --core-only
) ELSE (
    python main.py
)

echo.
echo Done! Results saved in results\
echo   Q1 Summary : results\q1_summary.txt
echo   LaTeX Table: results\latex_table.tex
pause
