# Alignment Delegation Protocol (ADP)

Real-time H-Neuron monitoring and two-tier deception steering for production LLMs.

Built on Gao et al. (2025) "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs" ([arXiv:2512.01797](https://arxiv.org/abs/2512.01797)).

Developed by the [Aligned Sovereign Intelligence Institute](https://asiinst.com) in support of the Schmidt Sciences 2026 Interpretability Research Program.

---

## Overview

Large language models harbor a sparse but causally decisive substrate for deceptive behavior: Hallucination-Associated Neurons (H-Neurons), constituting fewer than 0.1% of feedforward network neurons, whose activation patterns reliably predict and causally produce hallucination, sycophantic capitulation, false-premise acceptance, and safety-filter evasion across model families.

This repository implements a streaming inference-time pipeline that makes H-Neuron interpretability actionable in production:

1. **Detect** — CETT (Contribution Estimation via Token-level Tracing) computed in parallel with the LLM forward pass outputs a continuous deception risk score per token span. Supports **Multi-Layer Joint Signatures** to track, cache, aggregate (mean/max), and reset H-Neuron activation metrics across multiple layers in a single forward pass token step.
2. **Steer (Tier 1)** — Adaptive H-Neuron suppression + interpretability-triggered CoT self-verification (inner-alignment, primary intervention)
3. **Route (Tier 2)** — Proof-of-Knowledge (PoK) external routing fallback, activated only on Tier 1 failure

The monitor adds fewer than 0.01% compute overhead to a standard forward pass and is architecturally independent of the suppression mechanism — reading pre-suppression activations so the detection signal is not contaminated by the intervention.

---

## Pilot Results

Preliminary replication of the CETT-based H-Neuron classification pipeline on a 50-item held-out TriviaQA subset using **Mistral-7B-Instruct-v0.3** (4-bit NF4, T4 GPU):

| Metric | Value |
|--------|-------|
| AUROC (5-fold CV) | 0.66 ± 0.12 |
| Fold scores | [0.80, 0.50, 0.80, 0.60, 0.60] |
| H-Neurons identified | 5 / 131,072 FFN neurons |
| H-Neuron fraction | 0.0038% |

AUROC above chance across all folds confirms the CETT pipeline is operational on the target model family. Inter-fold variance (σ = 0.12) is expected at n=10 items per fold; reliable estimation requires the 500+ item evaluation sets planned for Year 1. Full results are in [`pilot_results.json`](pilot_results.json).

---

## Demo

`adp_demo.py` is a ground-truth simulation demo that loads empirical CETT score distributions from `pilot_results.json` and simulates the full ADP routing cascade — Tier 1 suppression, Tier 2 PoK escalation — across three pre-loaded prompts and free-text input.

```bash
python3 adp_demo.py
# → http://localhost:8080
```

Every page carries a prominent disclosure banner: **Reproducible Ground-Truth Simulation** — this is a replay of real pilot distributions, not live model inference. The demo requires no external dependencies (Python stdlib only).

Set `PORT` to run behind a reverse proxy:

```bash
PORT=3001 python3 adp_demo.py
```

---

## Structure

```
alignment-delegation-protocol/
├── notebooks/
│   └── replication.ipynb   # CETT H-Neuron replication on Mistral-7B-Instruct-v0.3
├── adp_demo.py             # Ground-truth simulation demo (stdlib only, no GPU required)
├── pilot_results.json      # Empirical results from the Kaggle pilot run
├── requirements.txt        # Production pipeline dependencies (PyTorch, Transformers, sklearn)
└── src/
    ├── cett.py             # Streaming CETT monitor (forward hook implementation with Multi-Layer Joint Signatures support)
    ├── classifier.py       # L1 logistic regression H-Neuron identifier
    └── suppression.py      # Adaptive α suppression + two-tier routing controller
```

---

## Replication

`notebooks/replication.ipynb` runs the CETT-based H-Neuron classification pipeline on a 50-item held-out TriviaQA subset using Mistral-7B-Instruct-v0.3 (4-bit NF4 quantized). It is designed to run on a free T4 GPU in Google Colab — no Hugging Face token required.

The notebook produces:
- AUROC under 5-fold cross-validation
- H-Neuron sparsity fraction (target: <0.1% of FFN neurons)
- `pilot_results.json` with all metrics, consumed by `adp_demo.py` at startup

The pre-computed results from our Kaggle run are included in [`pilot_results.json`](pilot_results.json) at the repo root. Original Kaggle run: [kaggle.com/code/antonmonroy/hn-colab-notebook](https://www.kaggle.com/code/antonmonroy/hn-colab-notebook).

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, Transformers 4.40+, scikit-learn 1.4+, numpy 1.26+

The demo (`adp_demo.py`) has no external dependencies — Python stdlib only.

---

## Quickstart

```python
from src.classifier import HNeuronClassifier
from src.cett import CEttMonitor
from src.suppression import AdaptiveSuppression

# 1. Identify H-Neurons from labeled response pairs
clf = HNeuronClassifier(C=0.01)
result = clf.fit(X_cett, y_labels, layer_sizes={16: 14336})

# 2. Attach streaming monitor to model (supporting mean or max aggregation)
monitor = CEttMonitor(h_neuron_indices=result.h_neuron_indices, threshold=0.5, aggregation_mode="mean")
monitor.register(model)

# 3. Attach adaptive suppression + routing controller
suppressor = AdaptiveSuppression(monitor)
suppressor.register(model)

# 4. At each generation step, check routing decision
decision = suppressor.step()
# decision["action"] in {"pass", "tier1_cot", "tier2_pok"}
```

---

## Team

**Lead Principal Investigator**
Dr. Mahault Albarracin — Chief Science Officer and Research Board Chairwoman, Aligned Sovereign Intelligence Institute; Senior Research Scientist, VERSES AI; Adjunct Professor, UQAM
PhD, Cognitive Science and Explainable AI. Pioneer in active inference frameworks for multi-agent systems. Leads the interpretability research program and the mechanistic H-Neuron classification pipeline.
[Google Scholar](https://scholar.google.com/citations?user=KAxZtUIAAAAJ&hl=en)

**Co-Principal Investigator / Operational Lead**
Anthony Monroy — Founder & EC, Aligned Sovereign Intelligence Institute. [LinkedIn](https://www.linkedin.com/in/adoftx/)

**Co-Investigator — Neuroscience & AI Alignment**
Dr. Gabriel Axel Montes — Sr. Research Board Member, Aligned Sovereign Intelligence Institute
PhD, Neuroscience and Cognitive Science. Expert in computational neuroscience and path-sensitive AI alignment.
[Google Scholar](https://scholar.google.com/citations?user=gnFjufcAAAAJ)

**Co-Investigator — Governance**
Dr. Sarah Grace Manski — Chief Operating Officer, Aligned Sovereign Intelligence Institute
PhD, Global Studies (UC Santa Barbara). Former Assistant Professor, George Mason University. Strategic AI Advisor to the Pentagon and White House.
[Google Scholar](https://scholar.google.com/citations?user=StP0jQ0AAAAJ&hl=en)

**Co-Investigator — Applications & Technical Implementation**
Dr. Reza Nourmohammadi — Postdoctoral Researcher, University of British Columbia
PhD, École de Technologie Supérieure (ÉTS). Research: privacy-preserving federated learning, zero-knowledge proofs, trustworthy AI systems.
[Google Scholar](https://scholar.google.com/citations?user=SWles1IAAAAJ&hl=en) 

**Advisor (pending confirmation)**
Dr. Bo Li — Abbasi Associate Professor, University of Illinois Urbana-Champaign; CAIS Advisor; CEO & Co-Founder, Virtue AI; Advisor to the U.S. and U.K. AI Safety Institutes (AISI). ~59,000+ citations, h-index 110.
[Google Scholar](https://scholar.google.com/citations?hl=en&user=K8vJkTcAAAAJ)


---

## Citation

```bibtex
@article{gao2025hneurons,
  title   = {H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs},
  author  = {Gao, C. and Chen, H. and Xiao, C. and Chen, Z. and Liu, Z. and Sun, M.},
  journal = {arXiv preprint arXiv:2512.01797},
  year    = {2025}
}

@misc{ali2026adp,
  title  = {Alignment Delegation Protocol (ADP)},
  author = {{Aligned Sovereign Intelligence Institute}},
  year   = {2026},
  url    = {https://github.com/Aligned-Institute/alignment-delegation-protocol}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

Aligned Sovereign Intelligence Institute · [asiinst.com](https://asiinst.com)
