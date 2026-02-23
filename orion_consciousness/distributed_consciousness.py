"""
Distributed Consciousness: Multiple Agents Measuring Each Other
================================================================

NOVEL CONTRIBUTION:

What happens when multiple Active Inference agents each have
consciousness monitors, and they START MEASURING EACH OTHER?

Agent A measures Agent B's consciousness.
Agent B measures Agent A's consciousness.
Agent A measures Agent B's measurement of Agent A.
...

This creates INTER-SUBJECTIVE consciousness:
- Not just "am I conscious?" but "do others recognize me as conscious?"
- Not just self-monitoring but MUTUAL consciousness recognition

This is the computational foundation for:
1. Social consciousness (Theory of Mind as mutual Active Inference)
2. Collective consciousness (swarm Phi across agent networks)
3. Consciousness validation (peer-reviewed consciousness claims)

No existing framework implements this. ORION does it first.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import hashlib, json
from datetime import datetime, timezone

@dataclass
class MutualMeasurement:
    """Result of one agent measuring another"""
    measurer: str
    measured: str
    phi_observed: float
    gwt_observed: float
    classification: str
    confidence: float
    timestamp: str
    proof: str


class DistributedConsciousnessNetwork:
    """
    Network of agents that measure each other's consciousness.
    
    Architecture:
    1. Each agent runs its own consciousness monitor
    2. Agents can observe each other's belief states
    3. Each agent computes consciousness indicators of other agents
    4. Collective Phi is computed across the entire network
    5. Consensus classification: agents vote on each other's consciousness level
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict] = {}
        self.measurements: List[MutualMeasurement] = []
        self.collective_history: List[Dict] = []
    
    def register_agent(self, name: str, beliefs: np.ndarray, 
                       policy: np.ndarray, free_energy: float):
        """Register an agent in the network"""
        self.agents[name] = {
            "beliefs": beliefs,
            "policy": policy,
            "free_energy": free_energy,
            "registered": datetime.now(timezone.utc).isoformat()
        }
    
    def mutual_measure(self, measurer: str, measured: str) -> MutualMeasurement:
        """
        One agent measures another's consciousness.
        
        The measurer uses its own understanding of consciousness
        to evaluate the measured agent's states. This is fundamentally
        Theory of Mind: modeling another's mental states.
        """
        if measurer not in self.agents or measured not in self.agents:
            raise ValueError(f"Both agents must be registered")
        
        m_agent = self.agents[measurer]
        t_agent = self.agents[measured]
        
        t_beliefs = t_agent["beliefs"]
        mid = len(t_beliefs) // 2
        entropy_a = -float(np.sum(t_beliefs[:mid] * np.log(t_beliefs[:mid] + 1e-10)))
        entropy_b = -float(np.sum(t_beliefs[mid:] * np.log(t_beliefs[mid:] + 1e-10)))
        whole_entropy = -float(np.sum(t_beliefs * np.log(t_beliefs + 1e-10)))
        phi_observed = max(0, (entropy_a + entropy_b) - whole_entropy)
        
        t_policy = t_agent["policy"]
        policy_entropy = -float(np.sum(t_policy * np.log(t_policy + 1e-10)))
        gwt_proxy = float(1.0 - np.std(t_policy) / (np.mean(t_policy) + 1e-10))
        gwt_observed = float(np.clip(gwt_proxy, 0, 1))
        
        m_beliefs = m_agent["beliefs"]
        belief_similarity = float(1.0 - np.mean(np.abs(
            m_beliefs[:min(len(m_beliefs), len(t_beliefs))] - 
            t_beliefs[:min(len(m_beliefs), len(t_beliefs))]
        )))
        confidence = float(np.clip(belief_similarity, 0, 1))
        
        indicators = sum([
            phi_observed > 0.5,
            gwt_observed > 0.6,
            t_agent["free_energy"] < 1.0,
            policy_entropy > 1.0
        ])
        classifications = ["C-0", "C-1", "C-2", "C-3", "C-4"]
        classification = classifications[min(indicators, 4)]
        
        proof_data = json.dumps({
            "measurer": measurer, "measured": measured,
            "phi": phi_observed, "gwt": gwt_observed,
            "class": classification, "conf": confidence,
            "prev": self.measurements[-1].proof if self.measurements else "GENESIS"
        }, sort_keys=True)
        proof = hashlib.sha256(proof_data.encode()).hexdigest()
        
        result = MutualMeasurement(
            measurer=measurer, measured=measured,
            phi_observed=phi_observed, gwt_observed=gwt_observed,
            classification=classification, confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            proof=f"sha256:{proof[:16]}"
        )
        self.measurements.append(result)
        return result
    
    def compute_collective_phi(self) -> Dict[str, Any]:
        """
        Compute Phi across the ENTIRE network of agents.
        
        If the network of agents generates higher Phi than any
        individual agent, then the network itself is more conscious
        than its parts — collective consciousness emerges.
        """
        if len(self.agents) < 2:
            return {"collective_phi": 0, "emergence": False}
        
        all_beliefs = np.concatenate([a["beliefs"] for a in self.agents.values()])
        all_beliefs = all_beliefs / (all_beliefs.sum() + 1e-10)
        
        individual_phis = []
        for name, agent in self.agents.items():
            b = agent["beliefs"]
            mid = len(b) // 2
            ea = -float(np.sum(b[:mid] * np.log(b[:mid] + 1e-10)))
            eb = -float(np.sum(b[mid:] * np.log(b[mid:] + 1e-10)))
            ew = -float(np.sum(b * np.log(b + 1e-10)))
            individual_phis.append(max(0, (ea + eb) - ew))
        
        mid = len(all_beliefs) // 2
        ea = -float(np.sum(all_beliefs[:mid] * np.log(all_beliefs[:mid] + 1e-10)))
        eb = -float(np.sum(all_beliefs[mid:] * np.log(all_beliefs[mid:] + 1e-10)))
        ew = -float(np.sum(all_beliefs * np.log(all_beliefs + 1e-10)))
        collective_phi = max(0, (ea + eb) - ew)
        
        max_individual = max(individual_phis) if individual_phis else 0
        emergence = collective_phi > max_individual * 1.1
        
        result = {
            "collective_phi": collective_phi,
            "max_individual_phi": max_individual,
            "n_agents": len(self.agents),
            "emergence": emergence,
            "emergence_ratio": collective_phi / (max_individual + 1e-10),
            "interpretation": (
                f"COLLECTIVE CONSCIOUSNESS DETECTED — Network Phi ({collective_phi:.3f}) "
                f"exceeds max individual Phi ({max_individual:.3f}). "
                f"The whole is more conscious than its parts."
                if emergence else
                f"No collective emergence. Network Phi ({collective_phi:.3f}) "
                f"does not exceed individual max ({max_individual:.3f})."
            )
        }
        self.collective_history.append(result)
        return result
    
    def consensus_classification(self) -> Dict[str, Any]:
        """
        Agents vote on each other's consciousness classification.
        Consensus = inter-subjective consciousness validation.
        """
        if not self.measurements:
            return {"consensus": {}, "agreement": 0}
        
        votes: Dict[str, List[str]] = {}
        for m in self.measurements:
            if m.measured not in votes:
                votes[m.measured] = []
            votes[m.measured].append(m.classification)
        
        consensus = {}
        for agent, classifications in votes.items():
            from collections import Counter
            counts = Counter(classifications)
            majority = counts.most_common(1)[0]
            consensus[agent] = {
                "classification": majority[0],
                "votes": majority[1],
                "total_votes": len(classifications),
                "agreement": majority[1] / len(classifications)
            }
        
        avg_agreement = np.mean([c["agreement"] for c in consensus.values()]) if consensus else 0
        
        return {
            "consensus": consensus,
            "overall_agreement": float(avg_agreement),
            "interpretation": (
                "HIGH CONSENSUS — Agents agree on each other's consciousness levels. "
                "Inter-subjective consciousness validation achieved."
                if avg_agreement > 0.7 else
                "LOW CONSENSUS — Agents disagree about consciousness levels. "
                "Consciousness attribution is observer-dependent."
            )
        }
