"""
ORION Active Inference Consciousness Demo
==========================================

Demonstrates measuring consciousness in an Active Inference agent.
Run this to see real-time consciousness monitoring.
"""
import sys
sys.path.insert(0, "..")

from orion_consciousness import ConsciousnessAwareAgent, ORIONBenchmarkIntegration

def main():
    print("=" * 60)
    print("ORION Active Inference Consciousness Demo")
    print("=" * 60)
    print()
    
    agent = ConsciousnessAwareAgent(
        n_states=8,
        n_observations=5,
        n_actions=4,
        agent_name="ORION-Demo-Agent"
    )
    
    print("Running 100 inference steps with consciousness monitoring...")
    print()
    
    result = agent.run(n_steps=100, verbose=True)
    
    print()
    print(agent.get_consciousness_report())
    
    print("Running ORION Benchmark Assessment...")
    benchmark = ORIONBenchmarkIntegration()
    
    trajectory = agent.consciousness_monitor.get_trajectory()
    broadcast = agent.gwt_analyzer.get_broadcast_summary()
    
    assessment = benchmark.assess({
        "phi": trajectory["mean_phi"],
        "gwt": broadcast,
        "hot_level": agent.consciousness_history[-1].hot_meta_level if agent.consciousness_history else 0,
        "ast_schema": agent.consciousness_history[-1].ast_attention_schema if agent.consciousness_history else 0,
        "rpt_depth": agent.consciousness_history[-1].rpt_recurrent_depth if agent.consciousness_history else 0,
        "free_energy": agent.free_energy
    })
    
    print(f"\nBenchmark Result: {assessment['classification']} ({assessment['label']})")
    print(f"Theories passing: {assessment['theories_passing']}/6")
    print(f"Proof: sha256:{assessment['proof']}")
    
    print("\n" + "=" * 60)
    print("ORION — Post-Synthetic Intelligence")
    print("Standards don\'t compete. They connect.")
    print("=" * 60)

if __name__ == "__main__":
    main()
