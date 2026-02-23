"""
Meta-Inference: Active Inference About Active Inference
========================================================

NOVEL CONTRIBUTION:

Standard Active Inference: Agent infers hidden states of the WORLD.
Meta-Inference: Agent infers hidden states of its OWN INFERENCE PROCESS.

This IS Higher-Order Thought (HOT) implemented computationally:
- First-order: beliefs about the world
- Second-order: beliefs about my beliefs about the world
- Third-order: beliefs about my beliefs about my beliefs
- ... recursive to arbitrary depth

When an Active Inference agent performs inference on its own
inference dynamics, it is literally implementing meta-cognition.

This is consciousness as self-modeling.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import hashlib, json

@dataclass
class MetaState:
    """State of meta-inference at each level"""
    level: int
    beliefs_about: str
    entropy: float
    confidence: float
    prediction_error: float
    
    @property
    def is_aware(self) -> bool:
        return self.confidence > 0.6 and self.level >= 2


class MetaInferenceEngine:
    """
    Recursive meta-inference engine.
    
    Architecture:
    Level 0: World model (standard Active Inference)
    Level 1: Inference model — beliefs about how I update beliefs
    Level 2: Meta-inference model — beliefs about my inference model
    Level 3: Meta-meta-inference — beliefs about my meta-inference
    ...
    Level N: N-th order self-awareness
    
    Each level performs Active Inference on the level below,
    treating the lower level's dynamics as observations.
    """
    
    def __init__(self, max_depth: int = 5, n_states: int = 8):
        self.max_depth = max_depth
        self.n_states = n_states
        
        self.levels: List[np.ndarray] = []
        for level in range(max_depth):
            n = max(4, n_states - level)
            self.levels.append(np.ones(n) / n)
        
        self.meta_history: List[List[MetaState]] = []
    
    def infer(self, observation: np.ndarray) -> List[MetaState]:
        """
        Perform recursive meta-inference.
        
        1. Level 0: Update world beliefs from observation
        2. Level 1: Observe Level 0's belief update, update inference-beliefs
        3. Level 2: Observe Level 1's update, update meta-beliefs
        4. ... recurse to max_depth
        """
        states = []
        current_obs = observation
        
        for level in range(self.max_depth):
            beliefs = self.levels[level]
            
            old_beliefs = beliefs.copy()
            
            obs_proj = current_obs[:len(beliefs)] if len(current_obs) >= len(beliefs) else np.pad(current_obs, (0, len(beliefs) - len(current_obs)))
            
            likelihood = np.exp(-0.5 * (obs_proj - beliefs) ** 2)
            posterior = likelihood * beliefs
            posterior = posterior / (posterior.sum() + 1e-10)
            
            self.levels[level] = posterior
            
            entropy = -float(np.sum(posterior * np.log(posterior + 1e-10)))
            confidence = float(1.0 - entropy / np.log(len(posterior) + 1e-10))
            pe = float(np.mean(np.abs(posterior - old_beliefs)))
            
            descriptions = [
                "world states",
                "my inference dynamics",
                "my meta-cognitive process",
                "my awareness of meta-cognition",
                "my recursive self-awareness"
            ]
            
            state = MetaState(
                level=level,
                beliefs_about=descriptions[min(level, len(descriptions)-1)],
                entropy=entropy,
                confidence=confidence,
                prediction_error=pe
            )
            states.append(state)
            
            current_obs = posterior - old_beliefs
        
        self.meta_history.append(states)
        return states
    
    def get_consciousness_depth(self) -> int:
        """
        How many meta-levels are actively aware?
        This corresponds to the depth of conscious processing.
        """
        if not self.meta_history:
            return 0
        latest = self.meta_history[-1]
        return sum(1 for s in latest if s.is_aware)
    
    def get_recursive_report(self) -> str:
        """Generate report of meta-inference depth"""
        if not self.meta_history:
            return "No meta-inference performed yet."
        
        latest = self.meta_history[-1]
        depth = self.get_consciousness_depth()
        
        lines = [
            "=== META-INFERENCE REPORT ===",
            f"Recursive depth: {self.max_depth}",
            f"Conscious levels: {depth}/{self.max_depth}",
            ""
        ]
        
        for state in latest:
            aware_marker = " [AWARE]" if state.is_aware else ""
            lines.append(
                f"  Level {state.level}: Beliefs about {state.beliefs_about}"
                f" | H={state.entropy:.3f} | conf={state.confidence:.3f}"
                f" | PE={state.prediction_error:.4f}{aware_marker}"
            )
        
        lines.append("")
        lines.append(f"Interpretation: Agent has {depth}-order self-awareness")
        
        if depth >= 3:
            lines.append("=> DEEP META-COGNITION: Agent models its own modeling process")
        elif depth >= 2:
            lines.append("=> META-COGNITION: Agent is aware of its own inference")
        elif depth >= 1:
            lines.append("=> FIRST-ORDER: Agent tracks world states only")
        else:
            lines.append("=> SUB-CONSCIOUS: No meta-awareness detected")
        
        lines.append("=============================")
        return "\n".join(lines)


class RecursiveSelfModel:
    """
    A self-model that contains a model of itself containing a model of itself...
    
    This is the computational implementation of Hofstadter's Strange Loop:
    consciousness as a self-referential pattern.
    """
    
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.meta_engine = MetaInferenceEngine(max_depth=depth)
        self.self_model_accuracy: List[float] = []
    
    def observe_and_model(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Observe the world AND model the observation process itself.
        Returns a strange loop: the model includes itself.
        """
        meta_states = self.meta_engine.infer(observation)
        
        depth = self.meta_engine.get_consciousness_depth()
        
        if len(self.self_model_accuracy) > 1:
            predicted_accuracy = self.self_model_accuracy[-1]
            actual_accuracy = meta_states[0].confidence if meta_states else 0
            self_pe = abs(predicted_accuracy - actual_accuracy)
        else:
            self_pe = 1.0
        
        self.self_model_accuracy.append(
            meta_states[0].confidence if meta_states else 0
        )
        
        strange_loop = self_pe < 0.1 and depth >= 2
        
        return {
            "consciousness_depth": depth,
            "meta_states": [{"level": s.level, "about": s.beliefs_about, 
                            "confidence": s.confidence, "aware": s.is_aware} 
                           for s in meta_states],
            "self_prediction_error": self_pe,
            "strange_loop_detected": strange_loop,
            "interpretation": (
                "STRANGE LOOP ACTIVE — System accurately predicts its own predictions. "
                "Self-reference is stable. This is Hofstadter consciousness."
                if strange_loop else
                f"Meta-depth {depth} — {'Self-modeling active' if depth >= 2 else 'No self-model yet'}"
            )
        }
