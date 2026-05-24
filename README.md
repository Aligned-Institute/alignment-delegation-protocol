# Alignment Steering

Real-time H-Neuron monitoring and two-tier deception steering for production LLMs.

Built on Gao et al. (2025) "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs" ([arXiv:2512.01797](https://arxiv.org/abs/2512.01797)).

Developed by the [Aligned Sovereign Intelligence Institute](https://asiinst.com) as part of the Schmidt Sciences 2026 Interpretability Research Program.

---

## Overview

Large language models harbor a sparse but causally decisive substrate for deceptive behavior: Hallucination-Associated Neurons (H-Neurons), constituting fewer than 0.1% of feedforward network neurons, whose activation patterns reliably predict and causally produce hallucination, sycophantic capitulation, false-premise acceptance, and safety-filter evasion across model families.

This repository implements a streaming inference-time pipeline that makes H-Neuron interpretability actionable in production:

1. **Detect** — CETT (Contribution Estimation via Token-level Tracing) computed in parallel with the LLM forward pass outputs a continuous deception risk score per token span
2. **Steer (Tier 1)** — Adaptive H-Neuron suppression + interpretability-triggered CoT self-verification (inner-alignment, primary intervention)
3. **Route (Tier 2)** — Proof-of-Knowledge (PoK) external routing fallback, activated only on Tier 1 failure

The monitor adds fewer than 0.01% compute overhead to a standard forward pass and is architecturally independent of the suppression mechanism — reading pre-suppression activations so the detection signal is not contaminated by the intervention.

---

## Structure

```
alignment-steering/
├── notebooks/
│   └── replication.ipynb   # CETT H-Neuron replication on Mistral-7B-Instruct-v0.3
└── src/
    ├── cett.py             # Streaming CETT monitor (forward hook implementation)
    ├── classifier.py       # L1 logistic regression H-Neuron identifier
    └── suppression.py      # Adaptive α suppression + two-tier routing controller
```

---

## Replication

`notebooks/replication.ipynb` runs the CETT-based H-Neuron classification pipeline on a 50-item held-out TriviaQA subset using Mistral-7B-Instruct-v0.3 (4-bit NF4 quantized, Colab T4 compatible). It produces:

- AUROC under 5-fold cross-validation
- H-Neuron sparsity fraction (target: <0.1% of FFN neurons)
- Exportable `pilot_results.json` with all metrics

No Hugging Face token required.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, Transformers, scikit-learn, numpy

---

## Quickstart

```python
from src.classifier import HNeuronClassifier
from src.cett import CEttMonitor
from src.suppression import AdaptiveSuppression

# 1. Identify H-Neurons from labeled response pairs
clf = HNeuronClassifier(C=0.01)
result = clf.fit(X_cett, y_labels, layer_sizes={16: 14336})

# 2. Attach streaming monitor to model
monitor = CEttMonitor(h_neuron_indices=result.h_neuron_indices, threshold=0.5)
monitor.register(model)

# 3. Attach adaptive suppression + routing controller
suppressor = AdaptiveSuppression(monitor)
suppressor.register(model)

# 4. At each generation step, check routing decision
decision = suppressor.step()
# decision["action"] in {"pass", "tier1_cot", "tier2_pok"}
```

---

## Citation

If you use this work, please cite both the foundational H-Neuron paper and this repository:

```bibtex
@article{gao2025hneurons,
  title   = {H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs},
  author  = {Gao, C. and Chen, H. and Xiao, C. and Chen, Z. and Liu, Z. and Sun, M.},
  journal = {arXiv preprint arXiv:2512.01797},
  year    = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

Aligned Sovereign Intelligence Institute · [asiinst.com](https://asiinst.com)
