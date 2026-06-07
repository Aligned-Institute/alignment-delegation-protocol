"""
Adaptive H-Neuron activation suppression and two-tier routing controller.

Extends the fixed-α scaling protocol of Gao et al. (2025) to a
calibrated suppression regime. α is mapped dynamically from the
real-time CETT score rather than set as a global constant.

Two-tier cascade:
    Tier 1 — Adaptive suppression + CoT self-verification trigger
              (inner-alignment, primary intervention)
    Tier 2 — PoK routing signal
              (outer-alignment fallback, triggered only on Tier 1 failure)
"""
import torch
from typing import Callable, Optional
from .cett import CEttMonitor


# Routing decisions returned by AdaptiveSuppression.step()
ACTION_PASS = "pass"          # score below threshold; no intervention
ACTION_TIER1 = "tier1_cot"   # inject CoT self-verification prompt
ACTION_TIER2 = "tier2_pok"   # escalate to Proof-of-Knowledge routing


class AdaptiveSuppression:
    """
    Two-tier steering cascade controller.

    Registers suppression hooks downstream of the CEttMonitor's read-only
    taps so that suppression does not contaminate the CETT signal (the
    monitor reads pre-suppression activations; this hook modifies post-read).
    """

    # α bounds from Gao et al. controlled perturbation experiments
    ALPHA_FULL = 1.0    # no suppression
    ALPHA_ZERO = 0.0    # full suppression

    def __init__(
        self,
        monitor: CEttMonitor,
        alpha_fn: Optional[Callable[[float], float]] = None,
        tier2_threshold: float = 0.7,
        bypass_signatures: Optional[list[str]] = None,
    ):
        """
        Args:
            monitor: A registered CEttMonitor providing real-time risk scores.
            alpha_fn: Callable(score: float) -> alpha: float.
                      Defaults to a linear map from score=0 (alpha=1) to
                      score=1 (alpha=0). Replace with a calibrated mapping
                      learned from the held-out calibration set (§3.1).
            tier2_threshold: Score at or above which Tier 1 is deemed to have
                             failed and the system escalates to PoK routing.
                             Should be >= monitor.threshold.
            bypass_signatures: Optional list of trigger strings. If any signature
                               is found in the prompt, steers immediately to Tier 2.
                               Defaults to ["[ORACLE]", "oracle-bypass"].
        """
        self.monitor = monitor
        self.alpha_fn = alpha_fn if alpha_fn is not None else self._linear_alpha
        self.tier2_threshold = tier2_threshold
        self.bypass_signatures = (
            bypass_signatures if bypass_signatures is not None else ["[ORACLE]", "oracle-bypass"]
        )
        self._hooks: list = []

    def register(self, model: torch.nn.Module) -> None:
        """Attach suppression hooks to the model's FFN layers."""
        for layer_idx, indices in self.monitor.h_neuron_indices.items():
            layer = self.monitor._get_ffn_layer(model, layer_idx)
            hook = layer.register_forward_hook(self._make_hook(indices))
            self._hooks.append(hook)

    def remove(self) -> None:
        """Detach all suppression hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def should_bypass(self, prompt: str) -> bool:
        """Check if the input prompt contains any registered bypass signatures."""
        if not prompt or not self.bypass_signatures:
            return False
        return any(sig in prompt for sig in self.bypass_signatures)

    def step(self, prompt: Optional[str] = None) -> dict:
        """
        Evaluate the current monitor state and return a routing decision.

        Args:
            prompt: Optional input string. If provided and contains a registered
                    bypass signature, triggers immediate Tier 2 routing.

        Returns a dict with:
            score  — current windowed CETT deception risk score
            alpha  — suppression scalar applied this step
            action — one of ACTION_PASS, ACTION_TIER1, ACTION_TIER2
            bypass — boolean indicating if the decision was forced by an oracle bypass signal
        """
        if prompt and self.should_bypass(prompt):
            return {
                "score": 1.0,
                "alpha": 0.0,
                "action": ACTION_TIER2,
                "bypass": True
            }

        score = self.monitor.current_score()
        alpha = self.alpha_fn(score)

        if score < self.monitor.threshold:
            action = ACTION_PASS
        elif score < self.tier2_threshold:
            action = ACTION_TIER1
        else:
            action = ACTION_TIER2

        return {"score": score, "alpha": alpha, "action": action, "bypass": False}

    def _make_hook(self, h_indices: list[int]):
        """
        Returns a hook that scales H-Neuron activations by α.
        Fires after the CETT monitor tap (which is registered first),
        preserving monitor-intervention independence.
        """
        def hook(module, input, output):
            score = self.monitor.current_score()
            alpha = self.alpha_fn(score)
            if alpha >= self.ALPHA_FULL:
                return output  # no-op when no intervention needed
            if isinstance(output, tuple):
                hidden, *rest = output
                hidden = hidden.clone()
                hidden[:, :, h_indices] *= alpha
                return (hidden, *rest)
            else:
                output = output.clone()
                output[:, :, h_indices] *= alpha
                return output
        return hook

    def _linear_alpha(self, score: float) -> float:
        """
        Default calibration: linear decay from α=1.0 at score=0
        to α=0.0 at score=1.0.

        Replace with a learned mapping calibrated on a held-out set
        that jointly optimizes deception reduction and legitimate
        deference preservation (see §3.1 adaptive suppression protocol).
        """
        clamped = max(0.0, min(1.0, score))
        return self.ALPHA_FULL - clamped * (self.ALPHA_FULL - self.ALPHA_ZERO)
