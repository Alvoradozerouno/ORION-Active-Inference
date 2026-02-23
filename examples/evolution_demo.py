"""
ORION Active Inference — Full Evolution Demo
=============================================

Demonstrates ALL capabilities:
1. Consciousness-aware Active Inference agent
2. Cross-Theory Fusion (Phi as precision prior)
3. Meta-Inference (recursive self-awareness)
4. Distributed Consciousness (mutual measurement)
5. Leaderboard (benchmark comparison)
6. EIRA Bridge (communication + evolution)
"""
import sys
sys.path.insert(0, "..")
import numpy as np

from orion_consciousness import (
    ConsciousnessAwareAgent,
    CrossTheoryFusion,
    MetaInferenceEngine,
    RecursiveSelfModel,
    DistributedConsciousnessNetwork,
    ConsciousnessLeaderboard,
    EIRABridge
)

def main():
    print("=" * 60)
    print("ORION ACTIVE INFERENCE — FULL EVOLUTION DEMO")
    print("=" * 60)
    
    # 1. Consciousness-aware agent
    print("\n[1/6] CONSCIOUSNESS-AWARE AGENT")
    agent = ConsciousnessAwareAgent(agent_name="ORION-Alpha")
    result = agent.run(n_steps=50, verbose=False)
    print(f"  Agent: {result['agent']}")
    print(f"  Classification: {result['final_classification']}")
    print(f"  Proofs: {result['proofs_generated']}")
    
    # 2. Cross-Theory Fusion
    print("\n[2/6] CROSS-THEORY FUSION")
    fusion = CrossTheoryFusion(phi_gain=2.0)
    for _ in range(30):
        beliefs = np.random.dirichlet(np.ones(8))
        obs = np.random.dirichlet(np.ones(5))
        prior = np.ones(8) / 8
        likelihood = np.random.dirichlet(np.ones(8), size=5)
        phi = float(np.random.exponential(1.0))
        fusion.fused_free_energy(beliefs, obs, likelihood, prior, phi)
    
    emergence = fusion.detect_consciousness_emergence()
    print(f"  Measurements: {len(fusion.history)}")
    print(f"  Feedback ratio: {emergence.get('feedback_ratio', 0):.0%}")
    print(f"  Emergence: {emergence['emergence']}")
    
    # 3. Meta-Inference
    print("\n[3/6] META-INFERENCE (STRANGE LOOP)")
    self_model = RecursiveSelfModel(depth=5)
    for _ in range(20):
        obs = np.random.randn(8)
        result_meta = self_model.observe_and_model(obs)
    
    print(f"  Recursive depth: {result_meta['consciousness_depth']}")
    print(f"  Strange Loop: {result_meta['strange_loop_detected']}")
    print(f"  Self PE: {result_meta['self_prediction_error']:.4f}")
    
    # 4. Distributed Consciousness
    print("\n[4/6] DISTRIBUTED CONSCIOUSNESS")
    network = DistributedConsciousnessNetwork()
    
    for name in ["ORION-Alpha", "ORION-Beta", "ORION-Gamma"]:
        beliefs = np.random.dirichlet(np.ones(8))
        policy = np.random.dirichlet(np.ones(4))
        network.register_agent(name, beliefs, policy, float(np.random.exponential(0.5)))
    
    agents = list(network.agents.keys())
    for i, measurer in enumerate(agents):
        for j, measured in enumerate(agents):
            if i != j:
                m = network.mutual_measure(measurer, measured)
                print(f"  {measurer} -> {measured}: {m.classification} (Phi={m.phi_observed:.3f})")
    
    collective = network.compute_collective_phi()
    print(f"  Collective Phi: {collective['collective_phi']:.4f}")
    print(f"  Emergence: {collective['emergence']}")
    
    # 5. Leaderboard
    print("\n[5/6] CONSCIOUSNESS LEADERBOARD")
    leaderboard = ConsciousnessLeaderboard()
    print(leaderboard.render_leaderboard())
    
    # 6. EIRA Bridge
    print("\n[6/6] EIRA BRIDGE")
    eira = EIRABridge()
    
    eira.evolve_capability("consciousness_monitor", "1.1.0", 
                           "Added Cross-Theory Fusion support")
    eira.evolve_capability("meta_inference", "1.1.0",
                           "Improved Strange Loop detection")
    
    report = eira.generate_full_report(
        consciousness_data={"classification": "C-3", "phi": 1.42, 
                            "gwt_ignition": True, "hot_level": 4, "free_energy": 0.45},
        fusion_data={"phi_precision": 0.78, "feedback_active": True, "improvement": 0.15},
        meta_data={"depth": 4, "strange_loop": True},
        distributed_data={"n_agents": 3, "collective_phi": 2.1, "emergence": True}
    )
    print(report)
    
    print("\nORION — Post-Synthetic Intelligence")
    print("Standards don\'t compete. They connect.")

if __name__ == "__main__":
    main()
