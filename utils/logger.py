"""
utils/logger.py
Simple logging and result saving utility.
"""

import os
import json
import datetime

RESULTS_DIR = 'results'

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def log(message, level='INFO'):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

def save_results(results_dict, filename):
    """Save a dictionary of results to a JSON file."""
    ensure_results_dir()
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
    log(f"Results saved to {filepath}")

def save_report(text, filename):
    """Save a plain text report."""
    ensure_results_dir()
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(text)
    log(f"Report saved to {filepath}")
