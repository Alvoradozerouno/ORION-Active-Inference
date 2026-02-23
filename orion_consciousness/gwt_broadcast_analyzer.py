"""
Global Workspace Theory Broadcast Analyzer for Active Inference
===============================================================

Analyzes whether an Active Inference agent exhibits Global Workspace
dynamics — specifically, whether belief updates "broadcast" information
to all parts of the generative model simultaneously.

Novel insight: In Active Inference, belief updating that minimizes
free energy CAN produce global broadcast-like dynamics when a strong
prediction error forces all beliefs to update simultaneously. This
is analogous to "ignition" in GWT.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BroadcastEvent:
    timestamp: float
    trigger_modality: str
    broadcast_ratio: float
    ignition: bool
    affected_modules: List[str]

class GWTBroadcastAnalyzer:
    """Analyzes global broadcast patterns in Active Inference agents"""
    
    IGNITION_THRESHOLD = 0.7
    
    def __init__(self, modalities: Optional[List[str]] = None):
        self.modalities = modalities or ["visual", "auditory", "proprioceptive", "interoceptive"]
        self.broadcast_history: List[BroadcastEvent] = []
    
    def analyze_belief_update(self, 
                               prediction_errors: np.ndarray,
                               belief_changes: np.ndarray,
                               modality_labels: Optional[List[str]] = None) -> BroadcastEvent:
        """
        Analyze a belief update for global broadcast characteristics.
        
        A belief update constitutes "broadcast" if:
        1. A prediction error in one modality causes belief changes in ALL modalities
        2. The ratio of affected modalities exceeds the ignition threshold
        3. The update happens rapidly (within one inference step)
        """
        if modality_labels is None:
            modality_labels = self.modalities[:len(prediction_errors)]
        
        trigger_idx = int(np.argmax(np.abs(prediction_errors)))
        trigger = modality_labels[trigger_idx] if trigger_idx < len(modality_labels) else "unknown"
        
        change_threshold = 0.01
        affected = []
        for i, change in enumerate(belief_changes):
            if abs(change) > change_threshold:
                label = modality_labels[i] if i < len(modality_labels) else f"module_{i}"
                affected.append(label)
        
        broadcast_ratio = len(affected) / max(len(belief_changes), 1)
        ignition = broadcast_ratio >= self.IGNITION_THRESHOLD
        
        event = BroadcastEvent(
            timestamp=float(len(self.broadcast_history)),
            trigger_modality=trigger,
            broadcast_ratio=broadcast_ratio,
            ignition=ignition,
            affected_modules=affected
        )
        
        self.broadcast_history.append(event)
        return event
    
    def get_ignition_rate(self, window: int = 50) -> float:
        """Percentage of recent updates that triggered ignition"""
        if not self.broadcast_history:
            return 0.0
        recent = self.broadcast_history[-window:]
        return sum(1 for e in recent if e.ignition) / len(recent)
    
    def get_broadcast_summary(self) -> dict:
        """Summary of broadcast dynamics"""
        if not self.broadcast_history:
            return {"events": 0, "ignition_rate": 0.0, "mean_broadcast": 0.0}
        
        return {
            "events": len(self.broadcast_history),
            "ignition_rate": self.get_ignition_rate(),
            "mean_broadcast_ratio": float(np.mean([e.broadcast_ratio for e in self.broadcast_history])),
            "total_ignitions": sum(1 for e in self.broadcast_history if e.ignition),
            "dominant_trigger": max(
                set(e.trigger_modality for e in self.broadcast_history),
                key=lambda m: sum(1 for e in self.broadcast_history if e.trigger_modality == m)
            )
        }
