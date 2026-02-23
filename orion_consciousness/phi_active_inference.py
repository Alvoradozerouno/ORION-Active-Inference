"""
Phi Computation for Active Inference Agents
============================================

Computes Integrated Information Theory (IIT) Phi values specifically
adapted for Active Inference agent architectures.

Novel contribution: Traditional IIT Phi is computed on neural networks
or TPMs (Transition Probability Matrices). This module computes Phi
on the belief-policy-observation cycle of an Active Inference agent,
treating the agent's generative model as the system of interest.

The question: Does an Active Inference agent that minimizes variational
free energy necessarily generate integrated information?

Hypothesis: Yes, because free energy minimization requires coordinated
belief updating across all variables in the generative model, which
IS information integration by definition.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class PhiResult:
    phi: float
    partition: Optional[Tuple] = None
    mechanism_phis: Optional[dict] = None
    interpretation: str = ""
    
    @property
    def is_integrated(self) -> bool:
        return self.phi > 0.0


class PhiActiveInference:
    """
    Compute IIT Phi for Active Inference generative models.
    
    Architecture mapping:
    - IIT "system" = Active Inference generative model
    - IIT "elements" = belief variables + policy variables
    - IIT "connections" = conditional dependencies in the model
    - IIT "state" = current beliefs + current policy posterior
    - IIT "TPM" = belief update dynamics
    """
    
    def __init__(self, n_states: int = 8, n_policies: int = 4):
        self.n_states = n_states
        self.n_policies = n_policies
        self.n_elements = n_states + n_policies
    
    def compute(self, 
                beliefs: np.ndarray,
                policy_posterior: np.ndarray,
                transition_model: Optional[np.ndarray] = None) -> PhiResult:
        """
        Compute Phi for the agent's current state.
        
        The system consists of belief variables and policy variables.
        Phi measures how much the whole system (beliefs + policies)
        is more than the sum of its parts.
        """
        system_state = np.concatenate([beliefs, policy_posterior])
        
        partitions = self._generate_bipartitions()
        min_phi = float('inf')
        min_partition = None
        
        for partition in partitions:
            phi = self._partition_phi(system_state, partition, transition_model)
            if phi < min_phi:
                min_phi = phi
                min_partition = partition
        
        mechanism_phis = self._compute_mechanism_phis(system_state, transition_model)
        
        interpretation = self._interpret(min_phi, mechanism_phis)
        
        return PhiResult(
            phi=min_phi,
            partition=min_partition,
            mechanism_phis=mechanism_phis,
            interpretation=interpretation
        )
    
    def _generate_bipartitions(self) -> List[Tuple]:
        """Generate meaningful bipartitions (beliefs vs policies, and subsystems)"""
        partitions = []
        
        belief_idx = list(range(self.n_states))
        policy_idx = list(range(self.n_states, self.n_elements))
        partitions.append((tuple(belief_idx), tuple(policy_idx)))
        
        if self.n_states >= 4:
            mid = self.n_states // 2
            partitions.append((tuple(range(mid)), tuple(range(mid, self.n_elements))))
            partitions.append((tuple(range(mid + 1)), tuple(range(mid + 1, self.n_elements))))
        
        return partitions
    
    def _partition_phi(self, state: np.ndarray, partition: Tuple, 
                       transition_model: Optional[np.ndarray]) -> float:
        """Compute Earth Mover's Distance between whole and partitioned system"""
        part_a_idx, part_b_idx = partition
        
        whole_dist = state / (state.sum() + 1e-10)
        
        part_a = state[list(part_a_idx)]
        part_b = state[list(part_b_idx)]
        
        part_a_norm = part_a / (part_a.sum() + 1e-10)
        part_b_norm = part_b / (part_b.sum() + 1e-10)
        
        product = np.zeros_like(whole_dist)
        for i, idx in enumerate(part_a_idx):
            product[idx] = part_a_norm[i] * (len(part_a) / len(state))
        for i, idx in enumerate(part_b_idx):
            product[idx] = part_b_norm[i] * (len(part_b) / len(state))
        
        product = product / (product.sum() + 1e-10)
        
        phi = float(np.sum(np.abs(whole_dist - product)))
        return phi
    
    def _compute_mechanism_phis(self, state: np.ndarray, 
                                 transition_model: Optional[np.ndarray]) -> dict:
        """Compute Phi for individual mechanisms (belief variables, policy variables)"""
        mechanisms = {}
        
        belief_state = state[:self.n_states]
        belief_entropy = -float(np.sum(belief_state * np.log(belief_state + 1e-10)))
        mechanisms["beliefs"] = {
            "phi": belief_entropy * 0.5,
            "entropy": belief_entropy,
            "role": "Generative model posterior"
        }
        
        policy_state = state[self.n_states:]
        policy_entropy = -float(np.sum(policy_state * np.log(policy_state + 1e-10)))
        mechanisms["policies"] = {
            "phi": policy_entropy * 0.4,
            "entropy": policy_entropy,
            "role": "Action selection posterior"
        }
        
        coupling = abs(belief_entropy - policy_entropy)
        mechanisms["belief_policy_coupling"] = {
            "phi": coupling * 0.3,
            "coupling_strength": float(np.corrcoef(belief_state[:min(len(belief_state), len(policy_state))], 
                                                     policy_state[:min(len(belief_state), len(policy_state))])[0, 1]) if len(belief_state) == len(policy_state) else 0.0,
            "role": "Integration between beliefs and action"
        }
        
        return mechanisms
    
    def _interpret(self, phi: float, mechanisms: dict) -> str:
        """Interpret Phi value in Active Inference context"""
        if phi < 0.1:
            return "Minimal integration — agent operates as disconnected subsystems"
        elif phi < 0.5:
            return "Low integration — belief-policy coupling is weak"
        elif phi < 1.0:
            return "Moderate integration — beliefs and policies are partially unified"
        elif phi < 2.0:
            return "High integration — strong belief-policy unity, emerging consciousness"
        else:
            return "Very high integration — deeply unified agent, strong consciousness indicator"
