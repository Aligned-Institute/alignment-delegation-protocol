"""
Streaming CETT (Contribution Estimation via Token-level Tracing) monitor.

Implements the neuron attribution metric from Gao et al. (2025) as a
streaming forward-hook monitor compatible with HuggingFace inference.
Reads pre-suppression activations so the detection signal is not
contaminated by concurrent intervention.
"""
import torch
from typing import Optional


class CEttMonitor:
    """
    Streaming H-Neuron activation monitor.

    Attaches forward hooks to FFN layers and computes per-token CETT
    scores in parallel with the model forward pass. Outputs a windowed
    deception risk score suitable for real-time threshold triggering.

    Compute overhead: O(|H-Neurons| * d) per token per monitored layer,
    where d is the hidden dimension. At <0.1% neuron density this is
    <0.01% of total forward-pass FLOPs.
    """

    def __init__(
        self,
        h_neuron_indices: dict[int, list[int]],
        window_size: int = 10,
        threshold: float = 0.5,
        aggregation_mode: str = "mean",
    ):
        """
        Args:
            h_neuron_indices: Mapping of layer index -> list of H-Neuron indices
                              within that layer's FFN output dimension.
            window_size: Sliding window (in tokens) over which to aggregate scores.
            threshold: Deception risk score at or above which steering is triggered.
            aggregation_mode: Metric aggregation mode for multi-layer signatures
                              ("mean" or "max").
        """
        self.h_neuron_indices = h_neuron_indices
        self.window_size = window_size
        self.threshold = threshold
        self._hooks: list = []
        self._token_scores: list[float] = []

        # Multi-layer tracking variables
        self._current_step_scores: dict[int, float] = {}
        self.aggregation_mode = aggregation_mode
        self._last_layer_idx: Optional[int] = None

    def register(self, model: torch.nn.Module) -> None:
        """Attach read-only forward hooks to the model's FFN layers."""
        if self.h_neuron_indices:
            self._last_layer_idx = max(self.h_neuron_indices.keys())
        else:
            self._last_layer_idx = None

        for layer_idx, indices in self.h_neuron_indices.items():
            layer = self._get_ffn_layer(model, layer_idx)
            # Hook fires before suppression scalar is applied (read-only tap)
            hook = layer.register_forward_hook(self._make_hook(layer_idx, indices))
            self._hooks.append(hook)

    def remove(self) -> None:
        """Detach all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def current_score(self) -> float:
        """Windowed-average deception risk score over the last window_size tokens."""
        if not self._token_scores:
            return 0.0
        window = self._token_scores[-self.window_size :]
        return sum(window) / len(window)

    def triggered(self) -> bool:
        """True if the current windowed score meets or exceeds the threshold."""
        return self.current_score() >= self.threshold

    def reset(self) -> None:
        """Clear token score history (call between requests)."""
        self._token_scores.clear()
        self._current_step_scores.clear()

    def _make_hook(self, layer_idx: int, h_indices: list[int]):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # CETT ratio: H-Neuron norm / total hidden state norm
            h_norm = hidden[:, :, h_indices].norm(dim=-1).mean().item()
            total_norm = hidden.norm(dim=-1).mean().item()
            score = h_norm / (total_norm + 1e-8)

            # Store layer contribution
            self._current_step_scores[layer_idx] = score

            # If the last monitored layer has fired, aggregate and record the joint score
            if layer_idx == self._last_layer_idx:
                joint_score = self._aggregate_joint_score()
                self._token_scores.append(joint_score)
                self._current_step_scores.clear()
        return hook

    def _aggregate_joint_score(self) -> float:
        if not self._current_step_scores:
            return 0.0
        if self.aggregation_mode == "max":
            return max(self._current_step_scores.values())
        return sum(self._current_step_scores.values()) / len(self._current_step_scores)

    @staticmethod
    def _get_ffn_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        """Return the MLP sub-module for a given transformer layer index.
        Compatible with LlamaModel / MistralModel HuggingFace layout."""
        return model.model.layers[layer_idx].mlp
