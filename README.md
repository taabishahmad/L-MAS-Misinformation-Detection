# L-MAS: Lightweight Multi-Agent System for Misinformation Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Code for the paper: **"L-MAS: A Lightweight Multi-Agent System for Cost-Sensitive 
> Misinformation Detection with Cross-Domain Robustness in Cyber-Surveillance Environments"**  
> Tabish Ahmad — International Islamic University Islamabad (IIUI)  


---

## Overview

L-MAS is a three-agent cooperative system for political misinformation detection 
that operates entirely on CPU without GPU requirements.

| Agent | Role |
|---|---|
| Detection Agent | TF-IDF (10K features) + Logistic Regression |
| Verification Agent | Semantic similarity + Speaker credibility + Linguistic patterns |
| Decision Agent | Fixed-weight fusion + Adaptive Gradient Boosting (holdout stacking) |

**Key results on LIAR benchmark (n = 1,267 test set):**
- F1 = 0.6429 | Recall = 0.7812 | ROC-AUC = 0.7106
- False negatives reduced 46.9% (228 → 121) vs. best single-agent baseline
- ~15 MB RAM | ~28 ms/sample | CPU-only | No GPU required

---

## Datasets

This repository does **not** include raw datasets due to licensing. 
Download them from the original sources:

- **LIAR**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **FakeNewsNet**: https://github.com/KaiDMML/FakeNewsNet
- **ISOT Fake News**: https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/

Place downloaded data in a `data/` folder in the project root.

---

## Installation
```bash
git clone https://github.com/taabishahmad/L-MAS-Misinformation-Detection
cd L-MAS-Misinformation-Detection
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, scikit-learn 1.8.0, NumPy 2.4.2, SciPy 1.17.0

---

## Usage
```bash
# Run full L-MAS pipeline
python main.py

# Run ablation study
python ablation.py

# Run statistical tests
python statistical_tests.py
```

---

## Reproducibility

All experiments use `seed = 42`. Results were produced on AMD64 CPU, Windows 11.

To reproduce the exact numbers from the paper:
```bash
python run_all_experiments.py --seed 42
```

---

## Results

Full results are in the `results/` folder as JSON files. 
See `results/mas_metrics.json` for primary L-MAS performance.

---

## Citation

If you use this code, please cite:
```bibtex
@article{ahmad2026lmas,
  title     = {L-MAS: A Lightweight Multi-Agent System for Cost-Sensitive 
               Misinformation Detection with Cross-Domain Robustness in 
               Cyber-Surveillance Environments},
  author    = {Ahmad, Tabish},
  journal   = {Information Processing and Management},
  year      = {2026},
  publisher = {Elsevier}
}
```

---

## Contact

**Tabish Ahmad**  
Department of Computer Science, IIUI, Islamabad, Pakistan  
tabish.bscs4969@student.iiu.pk
tabishahmad871@hotmail.com
```

