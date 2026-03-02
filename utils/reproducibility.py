"""
utils/reproducibility.py

Reproducibility Utility
=======================
Sets all global random seeds and prints complete environment information.
Q1 journals increasingly require full reproducibility disclosure.

Usage:
    from utils.reproducibility import set_all_seeds, print_environment_info
    set_all_seeds(42)
    print_environment_info()
"""

import os
import sys
import random
import platform
import importlib
import numpy as np


GLOBAL_SEED = 42


def set_all_seeds(seed=GLOBAL_SEED):
    """
    Set all random seeds for full reproducibility.
    Must be called at the very start of any experiment script.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Reproducibility] All seeds set to {seed}.")
    return seed


def get_package_version(pkg_name):
    """Safely get installed version of a package."""
    try:
        mod = importlib.import_module(pkg_name.replace("-", "_"))
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return "NOT INSTALLED"


def print_environment_info(save_path="results/environment_info.txt"):
    """
    Print and save full environment details for reproducibility.

    Reports:
    - Python version
    - OS and hardware
    - All key package versions
    - Random seed setting
    - Dataset paths
    """
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 60)
    log("  ENVIRONMENT & REPRODUCIBILITY INFORMATION")
    log("=" * 60)

    log(f"\n  Python Version   : {sys.version}")
    log(f"  Platform         : {platform.system()} {platform.release()}")
    log(f"  Architecture     : {platform.machine()}")
    log(f"  Processor        : {platform.processor() or 'Unknown'}")

    log(f"\n  Random Seed      : {GLOBAL_SEED}")
    log(f"  PYTHONHASHSEED   : {os.environ.get('PYTHONHASHSEED', 'not set')}")

    log(f"\n  --- Package Versions ---")
    packages = [
        "sklearn", "pandas", "numpy", "matplotlib",
        "seaborn", "scipy", "joblib",
    ]
    for pkg in packages:
        v = get_package_version(pkg)
        display = pkg if pkg != "sklearn" else "scikit-learn"
        log(f"  {display:20s}: {v}")

    log(f"\n  --- Dataset Paths ---")
    log(f"  LIAR train.tsv   : data/raw/train.tsv")
    log(f"  LIAR valid.tsv   : data/raw/valid.tsv")
    log(f"  LIAR test.tsv    : data/raw/test.tsv")
    log(f"  Source           : https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")

    log(f"\n  --- Model Configuration ---")
    log(f"  Detection Agent  : TF-IDF(max_features=10000, ngram_range=(1,2))")
    log(f"                     LogisticRegression(C=1.0, class_weight='balanced')")
    log(f"  Verification     : Cosine Similarity + Speaker Credibility + Regex NER")
    log(f"  Decision Fusion  : alpha=0.55, beta=0.35, gamma=0.10, threshold=0.50")
    log(f"  Adaptive Fusion  : GradientBoosting(n_estimators=200, lr=0.05)")
    log("=" * 60)

    # Save to file
    os.makedirs("results", exist_ok=True)
    if save_path:
        with open(save_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n[Reproducibility] Saved → {save_path}")

    return "\n".join(lines)


def get_reproducibility_statement():
    """
    Returns a ready-to-paste reproducibility statement for the paper.
    """
    return (
        "To ensure reproducibility, all experiments use a fixed random seed of 42 "
        "applied to Python's random module, NumPy, and scikit-learn estimators "
        "via the random_state parameter. The LIAR dataset official train/validation/test "
        "splits are preserved as-is, with validation merged into training. "
        "No data augmentation or leakage between splits occurs. "
        "The complete implementation, including all agent classes, evaluation scripts, "
        "and preprocessing utilities, is available upon request to the corresponding author."
    )


if __name__ == "__main__":
    set_all_seeds(GLOBAL_SEED)
    print_environment_info()
    print("\n--- Paper Reproducibility Statement ---")
    print(get_reproducibility_statement())
