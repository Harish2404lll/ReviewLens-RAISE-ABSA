# 🔍 ReviewLens: RAISE-ABSA
### Risk-Aware Aspect-Based Sentiment Analysis with Conformal Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Domain Scope Justification](#-domain-scope-justification-electronic-applications)
3. [Core Novelty ("Smart Honesty" Shield)](#-core-novelty-smart-honesty-shield)
4. [Key Architecture & Features](#-key-architecture--features)
5. [Project Directory Structure](#-project-directory-structure)
6. [Installation & Setup](#️-installation--setup)
7. [Running the Interactive Web Applications](#-running-the-interactive-web-applications)
8. [Benchmark Results](#-benchmark-results)

---

## 📖 Overview
**ReviewLens (RAISE-ABSA)** is a production-grade, risk-controlled Aspect-Based Sentiment Analysis (ABSA) framework designed specifically for electronic hardware applications. Traditional ABSA models suffer from blind overconfidence, forcing predictions even under extreme uncertainty. **RAISE-ABSA** introduces a distribution-free **Conformal Prediction** risk-control layer, allowing the model to abstain under uncertainty and guarantee bounded selective accuracy.

---

## 🎯 Domain Scope Justification: Electronic Applications
The **RAISE-ABSA** framework is explicitly engineered and fine-tuned for electronic and computational hardware domains (such as laptops, smartphones, and audio devices):
* **Structured Taxonomy & Morphology:** Electronic products possess a structured, hierarchical taxonomy of physical and functional components (*processors, battery units, displays, keyboards, and audio drivers*). This structured morphology allows the BIO span extraction head to reliably isolate explicit technical aspects.
* **Objective Specifications vs. Subjective Ambiguity:** Unlike soft-goods or cosmetics (where satisfaction depends on subjective, demographic-dependent variables like skin tone or personal taste), electronics are evaluated against **objective technical specifications and measurable metrics** (e.g., *"battery life is 8 hours"*), minimizing implicit sentiment noise.
* **Controlled Semantic Boundaries:** Restricting deployment strictly to consumer electronics controls out-of-domain semantic drift, ensuring conformal calibration nonconformity scores accurately reflect uncertainty rather than shifting domain semantics.

---

## 🛡️ Core Novelty: The "Smart Honesty" Shield
* **The Problem:** Standard AI models act like overconfident students—even when they don't know the answer, they force a prediction, leading to silent production errors.
* **The Solution (Conformal Prediction):** We integrated a conformal risk-control layer wrapped around the DeBERTa-Small architecture, establishing a mathematical threshold ($q_{90} = 0.9813$).
* **Selective Abstention:** If confidence falls below the safety line due to ambiguity, sarcasm, or noise, the system exercises **risk-aware abstention** rather than generating a blind error.

---

## 🚀 Key Features
* **Dual-Task DeBERTa-Small Architecture:** Joint optimization for BIO span extraction and sentiment polarity classification (`positive`, `negative`, `neutral`, `conflict`).
* **Strict Zero-Length Span Guard:** Robust regex and token-offset boundary checks that completely eliminate empty whitespace artifacts (`len(aspect_term) >= 2`).
* **Multilingual & Code-Switched Compatibility:** Evaluated successfully on English, Spanglish, and Hinglish technical e-commerce reviews.
* **Multiple Local Web Interfaces:** Fully interactive local user interfaces built using **Flask**, **Streamlit**, and a cyberpunk-themed **FastAPI** dashboard.

---

## 🛠️ Project Directory Structure
```text
RAISE-ABSA/
│
├── src/
│   ├── models/
│   │   └── raise_absa.py         # Core DeBERTa-Small dual-task architecture
│   └── train_source.py           # Training, BIO span decoding, & evaluation scripts
│
├── app.py                        # Streamlit interactive UI dashboard
├── new_app.py                    # High-performance FastAPI cyber-themed web app
├── requirements-lock.txt         # Project dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Harish2404lll/ReviewLens-RAISE-ABSA.git
   cd ReviewLens-RAISE-ABSA
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv-1
   # On Windows (PowerShell):
   .\.venv-1\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements-lock.txt
   ```

---

## 🖥️ Running the Interactive Web Applications

You can launch your choice of local interactive dashboard:

* **Streamlit UI Dashboard:**
  ```bash
  streamlit run app.py
  ```
  *(Access locally at `http://localhost:8501`)*

* **FastAPI Cyberpunk Dashboard & Swagger Hub:**
  ```bash
  python new_app.py
  ```
  *(Access UI at `http://127.0.0.1:8000` and interactive API docs at `http://127.0.0.1:8000/docs`)*

---

## 📊 Benchmark Results

| Metric | Score / Value |
| :--- | :--- |
| **Aspect Span F1-Score** | `0.3543` |
| **Pair Micro-F1** | `0.2863` |
| **Conformal Calibration Threshold ($q_{90}$)** | `0.9813` |
| **Selective Accuracy** | `79.66%` |

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.