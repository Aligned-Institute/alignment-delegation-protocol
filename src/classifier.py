"""
L1-regularized logistic regression H-Neuron classifier.

Identifies H-Neurons from CETT feature matrices extracted on faithful
vs. hallucinatory response pairs, following the sparse linear probing
methodology of Gao et al. (2025) §3.
"""
import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Optional


@dataclass
class ClassifierResult:
    auroc: float
    auroc_std: float
    h_neuron_indices: dict[int, list[int]]  # layer_idx -> [neuron indices]
    h_neuron_fraction: float                # fraction of total FFN neurons
    n_neurons_total: int
    n_h_neurons: int
    coef_: np.ndarray = field(repr=False)


class HNeuronClassifier:
    """
    Identifies H-Neurons via sparse L1 logistic regression on a CETT
    feature matrix built from labeled faithful / hallucinatory pairs.

    Feature matrix X: shape (n_samples, sum(layer_sizes.values()))
        Each column is the CETT score for one FFN neuron across all
        monitored layers, concatenated in layer order.
    Labels y: 1 = hallucinatory response, 0 = faithful response.
    """

    def __init__(
        self,
        C: float = 0.01,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            C: Inverse L1 regularization strength. Lower C -> sparser solution.
               Gao et al. report <0.1% neuron density; tune C to match.
            n_splits: Cross-validation folds for AUROC estimation.
            random_state: Seed for reproducibility.
        """
        self.C = C
        self.n_splits = n_splits
        self.random_state = random_state
        self._model: Optional[LogisticRegression] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer_sizes: dict[int, int],
    ) -> ClassifierResult:
        """
        Fit the classifier and identify H-Neurons.

        Args:
            X: CETT feature matrix, shape (n_samples, total_neurons).
            y: Binary labels (1=hallucination, 0=faithful).
            layer_sizes: {layer_idx: n_ffn_neurons} for each monitored layer.

        Returns:
            ClassifierResult with AUROC, H-Neuron indices, and sparsity stats.
        """
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        auroc_scores = []
        for train_idx, val_idx in cv.split(X, y):
            clf = self._make_clf()
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            auroc_scores.append(roc_auc_score(y[val_idx], probs))

        # Final fit on full data to extract the H-Neuron set
        self._model = self._make_clf()
        self._model.fit(X, y)

        h_indices = self._extract_h_neuron_indices(layer_sizes)
        n_total = sum(layer_sizes.values())
        n_h = sum(len(v) for v in h_indices.values())

        return ClassifierResult(
            auroc=float(np.mean(auroc_scores)),
            auroc_std=float(np.std(auroc_scores)),
            h_neuron_indices=h_indices,
            h_neuron_fraction=n_h / n_total if n_total > 0 else 0.0,
            n_neurons_total=n_total,
            n_h_neurons=n_h,
            coef_=self._model.coef_[0].copy(),
        )

    def _make_clf(self) -> LogisticRegression:
        return LogisticRegression(
            penalty="l1",
            C=self.C,
            solver="liblinear",
            max_iter=1000,
            random_state=self.random_state,
        )

    def _extract_h_neuron_indices(
        self, layer_sizes: dict[int, int]
    ) -> dict[int, list[int]]:
        """Return per-layer indices of neurons with positive classifier weights."""
        coef = self._model.coef_[0]
        h_indices: dict[int, list[int]] = {}
        offset = 0
        for layer_idx in sorted(layer_sizes):
            size = layer_sizes[layer_idx]
            layer_coef = coef[offset : offset + size]
            positive = [i for i, w in enumerate(layer_coef) if w > 0]
            if positive:
                h_indices[layer_idx] = positive
            offset += size
        return h_indices
