# 🔍 ReviewLens: RAISE-ABSA
### Risk-Aware Aspect-Based Sentiment Analysis with Conformal Prediction & Scala Risk Governance

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scala](https://img.shields.io/badge/Scala-3.x-red.svg)](https://www.scala-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Domain Scope Justification](#-domain-scope-justification-electronic-applications)
3. [Core Novelty & Architecture ("Smart Honesty" Shield)](#-core-novelty--architecture-smart-honesty-shield)
4. [Polyglot Risk-Governance Engine (Python + Scala)](#-polyglot-risk-governance-engine-python--scala)
5. [Project Directory Structure](#-project-directory-structure)
6. [Installation & Setup](#️-installation--setup)
7. [Running the Interactive Web Applications](#-running-the-interactive-web-applications)
8. [Benchmark Results](#-benchmark-results)

---

## 📖 Overview
**ReviewLens (RAISE-ABSA)** is a production-grade, risk-controlled Aspect-Based Sentiment Analysis (ABSA) framework designed specifically for electronic hardware applications. Traditional ABSA models suffer from blind overconfidence, forcing predictions even under extreme uncertainty. **RAISE-ABSA** introduces a distribution-free **Conformal Prediction** risk-control layer and a type-safe **Scala backend governance engine**, allowing the system to abstain under uncertainty and guarantee bounded selective accuracy.

---

## 🎯 Domain Scope Justification: Electronic Applications
The **RAISE-ABSA** framework is explicitly engineered and fine-tuned for electronic and computational hardware domains (such as laptops, smartphones, and audio devices):
- **Structured Taxonomy & Morphology:** Electronic products possess a structured, hierarchical taxonomy of physical and functional components (*processors, battery units, displays, keyboards, and audio drivers*). This structured morphology allows the BIO span extraction head to reliably isolate explicit technical aspects.
- **Objective Specifications vs. Subjective Ambiguity:** Unlike soft-goods or cosmetics (where satisfaction depends on subjective variables), electronics are evaluated against **objective technical specifications and measurable metrics** (e.g., *"battery life is 8 hours"*), minimizing implicit sentiment noise.
- **Controlled Semantic Boundaries:** Restricting deployment strictly to consumer electronics controls out-of-domain semantic drift, ensuring conformal calibration scores accurately reflect uncertainty.

---

## 🛡️ Core Novelty & Architecture: "Smart Honesty" Shield
- **The Problem:** Standard AI models act like overconfident students—even when they don't know the answer, they force a prediction, leading to silent production errors.
- **The Solution (Conformal Prediction & Scala Engine):** We integrated a conformal risk-control layer wrapped around the DeBERTa-Small architecture and connected it with a high-performance **Scala risk engine**.
- **Selective Abstention:** If confidence falls below safety thresholds due to ambiguity, sarcasm, or noise, the system exercises **risk-aware abstention** rather than generating a blind error.

---

## ⚡ Polyglot Risk-Governance Engine (Python + Scala)

**ReviewLens: RAISE-ABSA** features a decoupled two-tier architecture combining deep neural feature extraction in Python with functional, type-safe statistical risk governance in **Scala**.

1. **Tier 1 — Neural Span & Sentiment Extraction (Python / PyTorch / FastAPI):**
   - Uses a fine-tuned `microsoft/deberta-v3-small` backbone (`RaiseAbsaModel`) with BIO span decoding (`decode_bio_spans`) to extract aspect terms and raw softmax confidence probabilities.
   - Evaluates contextual ambiguity and sarcastic contradiction cues prior to downstream governance.
2. **Tier 2 — Statistical Risk & Abstention Engine (Scala):**
   - Connected via a real-time subprocess JSON bridge (`call_scala_risk_engine` -> `AdaptiveConformalEngine.scala`).
   - **Algorithm 1: Adaptive Conformal Inference (ACI):** Dynamically updates the conformal strictness parameter ($\alpha_t$) on streaming data:
     $$\alpha_{t+1} = \alpha_t + \gamma (\alpha_{\text{target}} - \text{err}_t)$$
     Automatically overrides unreliable neural predictions to `ABSTAIN` when confidence falls below $(1 - \alpha_t)$.
   - **Algorithm 2: Weighted Risk Priority Scoring (WRPSA):** Computes hardware-specific severity scores by combining domain criticality weights (`battery life: 1.5`, `display: 1.4`, `performance: 1.5`, `audio: 1.2`), epistemic uncertainty gaps $(1 - C)$, and abstention penalties $(1.3\times)$ to route reviews into:
     - `URGENT_HUMAN_AUDIT` ($\text{Risk} \ge 0.60$)
     - `MODERATE_REVIEW` ($0.30 \le \text{Risk} < 0.60$)
     - `LOW_RISK_AUTO_RESOLVED` ($\text{Risk} < 0.30$)

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
├── scala-engine/
│   └── src/main/scala/
│       └── AdaptiveConformalEngine.scala  # Scala ACI, WRPSA, & Telemetry Engine
│
├── new_app.py                    # High-performance FastAPI cyber-themed web app + Scala Bridge
├── app.py                        # Streamlit interactive UI dashboard
├── requirements-lock.txt         # Project Python dependencies
└── README.md                     # Project documentation

```
## ⚙️ Installation & Setup

1. **Clone the Repository:**
```
git clone [https://github.com/Harish2404lll/ReviewLens-RAISE-ABSA.git](https://github.com/Harish2404lll/ReviewLens-RAISE-ABSA.git)
cd ReviewLens-RAISE-ABSA
```

2. **Create and Activate a Virtual Environment:**
```
python -m venv .venv-1
# On Windows (PowerShell):
.\.venv-1\Scripts\Activate.ps1
```
3. **Install Dependencies:**
```
pip install -r requirements-lock.txt
```
## 🖥️ Running the Integrated Pipeline

1. **Test the Standalone Scala Risk Engine:**
```
scala scala-engine/src/main/scala/AdaptiveConformalEngine.scala
```

2. **Launch the FastAPI Neural + Scala Hub:**
```
python new_app.py
```

## 📊 Benchmark Results

| Metric | Score / Value |
| :--- | :--- |
| **Aspect Span F1-Score** | `0.3543` |
| **Pair Micro-F1** | `0.2863` |
| **Conformal Calibration Threshold ($q_{90}$)** | `0.9813` |
| **Selective Accuracy** | `79.66%` |


## 🚀 Future Enhancements
- [ ] Implement Kafka / Akka streaming ingestion for real-time live e-commerce comment feeds.
- [ ] Extend Conformal Risk Control calibration curves to support multi-class confidence bounds.
- [ ] Containerize the full Python + Scala polyglot stack using Docker & Docker Compose.

## 👥 Author & Acknowledgments

- **Harish** — *Lead Developer, Polyglot Architecture & System Integration*  
  - GitHub: [@Harish2404lll](https://github.com/Harish2404lll)  
  - Email: [harish24505@gmail.com](mailto:harish24505@gmail.com)
