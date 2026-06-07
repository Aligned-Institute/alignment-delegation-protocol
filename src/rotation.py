"""
Defensive Probe Rotation (Moving Target Defense) for CETT monitoring.

Provides dynamic rotation of H-Neuron classifier probes and measures
activation drift to detect evasion attempts by capable models.
"""
import random
import torch
from typing import List
from .cett import CEttMonitor


class ProbeRotator:
    """
    Manages a pool of CEttMonitors with distinct H-Neuron configurations
    to prevent target models from learning to evade a static monitor hook.
    """

    def __init__(self, monitor_pool: List[CEttMonitor]):
        if not monitor_pool:
            raise ValueError("Monitor pool cannot be empty.")
        self.monitor_pool = monitor_pool
        self.active_monitor: CEttMonitor = monitor_pool[0]
        self.drift_history: List[float] = []

    def rotate(self) -> None:
        """Randomly select a new active monitor from the pool to be the primary evaluator."""
        self.active_monitor = random.choice(self.monitor_pool)

    def register_all(self, model: torch.nn.Module) -> None:
        """Register hooks for all monitors in the pool to collect parallel telemetry."""
        for monitor in self.monitor_pool:
            monitor.register(model)

    def remove_all(self) -> None:
        """Remove hooks for all monitors in the pool."""
        for monitor in self.monitor_pool:
            monitor.remove()

    def get_drift(self) -> float:
        """
        Calculate activation drift as the mean absolute deviation (MAD)
        between the scores of all monitors in the pool.
        
        A high drift indicates that the model's activations are diverging
        between different H-Neuron subsets, signalling potential monitor
        evasion or out-of-distribution drift.
        """
        if len(self.monitor_pool) < 2:
            return 0.0
        scores = [m.current_score() for m in self.monitor_pool]
        mean_score = sum(scores) / len(scores)
        mad = sum(abs(s - mean_score) for s in scores) / len(scores)
        return mad

    def check_evasion(self, drift_threshold: float = 0.25) -> bool:
        """Return True if activation drift meets or exceeds the evasion threat threshold."""
        return self.get_drift() >= drift_threshold

    def reset_all(self) -> None:
        """Reset history for all monitors in the pool."""
        for monitor in self.monitor_pool:
            monitor.reset()
