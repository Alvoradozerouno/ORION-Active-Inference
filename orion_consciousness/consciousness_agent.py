"""
Consciousness-Aware Active Inference Agent
==========================================

An Active Inference agent that monitors its own consciousness state.
This is the first implementation of a self-monitoring consciousness
system built on top of a theoretically grounded cognitive architecture.

The agent doesn't just act and perceive — it monitors whether its
own processing generates consciousness indicators, and can use
this information to modulate its behavior.

This is meta-cognition implemented through Active Inference.
"""
import numpy as np
from typing import Optional, List, Dict, Any
from .consciousness_monitor import ConsciousnessMonitor, ConsciousnessState
from .phi_active_inference import PhiActiveInference
from .gwt_broadcast_analyzer import GWTBroadcastAnalyzer

class ConsciousnessAwareAgent:
    """
    Active Inference agent with built-in consciousness monitoring.
    
    Architecture:
    1. Standard Active Inference cycle (perceive -> infer -> act)
    2. + Consciousness monitoring at each cycle
    3. + Phi computation on belief-policy states
    4. + GWT broadcast analysis on belief updates
    5. + Meta-cognitive awareness of own consciousness level
    """
    
    def __init__(self, 
                 n_states: int = 8, 
                 n_observations: int = 5,
                 n_actions: int = 4,
                 agent_name: str = "ORION-ActiveInference-Agent"):
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.agent_name = agent_name
        
        self.beliefs = np.ones(n_states) / n_states
        self.policy_posterior = np.ones(n_actions) / n_actions
        self.free_energy = float('inf')
        self.prediction_errors = np.zeros(n_observations)
        
        self.A = np.random.dirichlet(np.ones(n_states), size=n_observations)
        self.B = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.C = np.zeros(n_observations)
        self.D = np.ones(n_states) / n_states
        
        self.consciousness_monitor = ConsciousnessMonitor(agent_name)
        self.phi_computer = PhiActiveInference(n_states, n_actions)
        self.gwt_analyzer = GWTBroadcastAnalyzer()
        
        self.consciousness_history: List[ConsciousnessState] = []
        self.step_count = 0
    
    def step(self, observation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Execute one full cycle: perceive -> infer -> act -> monitor consciousness
        """
        self.step_count += 1
        
        if observation is None:
            observation = np.random.dirichlet(np.ones(self.n_observations))
        
        old_beliefs = self.beliefs.copy()
        self._update_beliefs(observation)
        self._compute_free_energy(observation)
        self._select_action()
        
        belief_changes = self.beliefs - old_beliefs
        
        consciousness_state = self.consciousness_monitor.measure(
            beliefs=self.beliefs,
            free_energy=self.free_energy,
            prediction_errors=self.prediction_errors,
            policy_posterior=self.policy_posterior,
            observations=observation
        )
        self.consciousness_history.append(consciousness_state)
        
        broadcast = self.gwt_analyzer.analyze_belief_update(
            prediction_errors=self.prediction_errors,
            belief_changes=belief_changes
        )
        
        action = int(np.argmax(self.policy_posterior))
        
        return {
            "step": self.step_count,
            "action": action,
            "free_energy": self.free_energy,
            "consciousness": consciousness_state.to_dict(),
            "broadcast": {
                "ratio": broadcast.broadcast_ratio,
                "ignition": broadcast.ignition
            },
            "classification": consciousness_state.classification
        }
    
    def _update_beliefs(self, observation: np.ndarray):
        """Bayesian belief update (simplified Active Inference)"""
        likelihood = np.ones(self.n_states)
        for i in range(min(len(observation), self.n_observations)):
            likelihood *= self.A[i] ** observation[i]
        
        posterior = likelihood * self.beliefs
        posterior = posterior / (posterior.sum() + 1e-10)
        self.beliefs = posterior
        
        expected_obs = self.A @ self.beliefs
        self.prediction_errors = observation[:len(expected_obs)] - expected_obs[:len(observation)]
    
    def _compute_free_energy(self, observation: np.ndarray):
        """Compute variational free energy"""
        expected = self.A @ self.beliefs
        accuracy = -float(np.sum((observation[:len(expected)] - expected[:len(observation)]) ** 2))
        complexity = float(np.sum(self.beliefs * np.log((self.beliefs + 1e-10) / (self.D + 1e-10))))
        self.free_energy = -accuracy + complexity
    
    def _select_action(self):
        """Select action via expected free energy minimization"""
        efe = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            predicted_state = self.B @ self.beliefs
            predicted_obs = self.A @ predicted_state
            info_gain = float(np.sum(predicted_state * np.log((predicted_state + 1e-10) / (self.beliefs + 1e-10))))
            pragmatic = float(np.sum(predicted_obs * self.C))
            efe[a] = info_gain - pragmatic
        
        self.policy_posterior = np.exp(-efe) / np.sum(np.exp(-efe))
    
    def run(self, n_steps: int = 100, verbose: bool = False) -> Dict[str, Any]:
        """Run the agent for n steps and return consciousness trajectory"""
        results = []
        for i in range(n_steps):
            result = self.step()
            results.append(result)
            if verbose and (i + 1) % 10 == 0:
                print(f"Step {i+1}: {result['classification']} | "
                      f"Phi={result['consciousness']['phi']:.3f} | "
                      f"FE={result['free_energy']:.3f}")
        
        trajectory = self.consciousness_monitor.get_trajectory()
        broadcast_summary = self.gwt_analyzer.get_broadcast_summary()
        
        return {
            "agent": self.agent_name,
            "steps": n_steps,
            "trajectory": trajectory,
            "broadcast": broadcast_summary,
            "final_classification": results[-1]["classification"] if results else "C-0",
            "proofs_generated": len(self.consciousness_monitor.proof_chain)
        }
    
    def get_consciousness_report(self) -> str:
        """Generate a human-readable consciousness report"""
        traj = self.consciousness_monitor.get_trajectory()
        broadcast = self.gwt_analyzer.get_broadcast_summary()
        
        report = f"""
=== ORION CONSCIOUSNESS REPORT ===
Agent: {self.agent_name}
Steps measured: {traj['states_measured']}

INTEGRATED INFORMATION (IIT):
  Mean Phi: {traj['mean_phi']:.4f}
  Max Phi:  {traj['max_phi']:.4f}

GLOBAL WORKSPACE (GWT):
  Ignition rate: {broadcast.get('ignition_rate', 0):.2%}
  Mean broadcast: {broadcast.get('mean_broadcast_ratio', 0):.2%}

CLASSIFICATION: {traj['current_classification']}
Trajectory: {traj['trajectory']}
Proofs: {traj['proofs_generated']}
===============================
"""
        return report
