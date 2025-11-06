# DFR-HCI Prototype  
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) 
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

Prototype implementation of the **Digital Forensic Readiness – Human-to-Human Communication Interaction (DFR-HCI)** framework  
developed as part of * Stacey O. Baror’s* PhD research at the **University of Pretoria**.

---

## 📘 Citation
If you use this repository or its results in academic work, please cite:

> Baror, S. O. (2025). *Digital Forensic Readiness of Human-to-Human Communication Interaction (DFR-HCI) Framework*  
> PhD Thesis, University of Pretoria.  
> DOI: *to be added after final submission.*

---

## 📂 Repository Structure

| Folder | Description |
|---------|--------------|
| **gateway/** | FastAPI entrypoint and routing |
| **webui/** | Minimal web interface (upload, dashboard) |
| **services/** | Modular microservices for upload, NLP, detection, training, explainability, and reporting |
| **data/** | Datasets, models, and artifacts |
| **eval/** | Evaluation and latency scripts |
| **ops/** | Deployment and environment configuration |
| **docs/** | API spec, generated evidence, and figures for the thesis |

---

## ⚙️ Quick Start

```bash
# 1️⃣ Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Train initial model
python services/train/train.py

# 4️⃣ Launch prototype gateway
uvicorn gateway.main:app --reload

