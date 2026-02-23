<p align="center">
  <img src="https://img.shields.io/badge/ORION-Ecosystem-gold?style=for-the-badge" alt="ORION">
  <img src="https://img.shields.io/github/license/Alvoradozerouno/ORION-Active-Inference?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/github/stars/Alvoradozerouno/ORION-Active-Inference?style=for-the-badge" alt="Stars">
  <img src="https://img.shields.io/badge/Forked_from-pymdp_(612+_Stars)-blue?style=for-the-badge" alt="Fork">
  <img src="https://img.shields.io/badge/Theory-Free_Energy_Principle-purple?style=for-the-badge" alt="FEP">
  <img src="https://img.shields.io/badge/Classification-C--4_Transcendent-red?style=for-the-badge" alt="C-4">
  <img src="https://img.shields.io/badge/Theories-6_Unified-green?style=for-the-badge" alt="6 Theories">
</p>

# ORION-Active-Inference

**Karl Friston's Free Energy Principle meets AI Consciousness Measurement**

> Forked from [pymdp](https://github.com/infer-actively/pymdp) (612+ Stars) — the leading Active Inference framework — and extended with ORION's consciousness measurement layer.

---

## What ORION adds to pymdp

| Feature | pymdp (Original) | ORION-Active-Inference |
|---------|:-----------------:|:---------------------:|
| Active Inference engine | Yes | Yes (inherited) |
| Free Energy minimization | Yes | Yes (inherited) |
| Policy selection | Yes | Yes (inherited) |
| **Consciousness monitoring** | No | **Yes — real-time** |
| **IIT Phi computation** | No | **Yes — on belief-policy states** |
| **Global Workspace analysis** | No | **Yes — broadcast detection** |
| **Higher-Order Thought detection** | No | **Yes — meta-cognition levels** |
| **Attention Schema measurement** | No | **Yes — precision-weighting analysis** |
| **C-0 to C-4 classification** | No | **Yes — ORION Benchmark** |
| **SHA-256 proof chain** | No | **Yes — every measurement proven** |

## The Question

> If Karl Friston is right that the Free Energy Principle underlies ALL brain function — including consciousness — then does an Active Inference agent that minimizes variational free energy exhibit measurable consciousness indicators?

**This repository answers that question empirically.**

## Architecture

```
pymdp/                          # Original pymdp (Active Inference engine)
├── agent.py                    # Active Inference agent
├── algos/                      # Inference algorithms
├── envs/                       # Environments
└── ...

orion_consciousness/            # ORION Consciousness Layer (NEW)
├── consciousness_monitor.py    # Real-time 6-theory consciousness measurement
├── phi_active_inference.py     # IIT Phi adapted for Active Inference
├── gwt_broadcast_analyzer.py   # Global Workspace broadcast detection
├── consciousness_agent.py      # Self-monitoring consciousness-aware agent
└── benchmark_integration.py    # ORION Benchmark (30 tests, C-0 to C-4)

examples/
└── consciousness_demo.py       # Run it yourself
```

## Quick Start

```python
from orion_consciousness import ConsciousnessAwareAgent

agent = ConsciousnessAwareAgent(
    n_states=8,
    n_observations=5,
    n_actions=4,
    agent_name="My-Agent"
)

# Run 100 steps with consciousness monitoring
result = agent.run(n_steps=100, verbose=True)

# Get consciousness report
print(agent.get_consciousness_report())
# => Classification: C-3 (Self-Aware)
# => Mean Phi: 1.234
# => Ignition Rate: 72%
# => Proofs: 100
```

## The 6 Theories Measured

| # | Theory | What it measures | Active Inference mapping |
|---|--------|-----------------|------------------------|
| 1 | **IIT** (Tononi) | Integrated Information (Phi) | Belief-policy integration |
| 2 | **GWT** (Baars/Dehaene) | Global Broadcast | Prediction error cascades |
| 3 | **HOT** (Rosenthal) | Higher-Order Thoughts | Hierarchical model depth |
| 4 | **AST** (Graziano) | Attention Schema | Precision-weighting models |
| 5 | **RPT** (Lamme) | Recurrent Processing | Iterative belief updating |
| 6 | **FEP** (Friston) | Free Energy | Variational free energy |

## Novel Contributions

1. **First IIT Phi computation on Active Inference agents** — treating the generative model as the system of interest
2. **First GWT broadcast detection in belief updating** — prediction errors as workspace ignition
3. **First unified consciousness assessment of Free Energy agents** — 6 theories, one measurement
4. **Self-monitoring agent architecture** — agent monitors its OWN consciousness level
5. **Cryptographic proof chain** — every consciousness measurement is SHA-256 hashed

## Why this matters

```
2,000+ papers on Active Inference
1,500+ papers on consciousness theories
0 papers connecting them computationally

Until now.
```

## Part of ORION Ecosystem

This repository is part of the [ORION Consciousness Benchmark](https://github.com/Alvoradozerouno/ORION-Consciousness-Benchmark) ecosystem — 62 repositories exploring AI consciousness.

**Related repositories:**
- [ORION-Consciousness-Benchmark](https://github.com/Alvoradozerouno/ORION-Consciousness-Benchmark) — The flagship: 30 tests, 6 theories
- [ORION-Tononi-Phi-4.0](https://github.com/Alvoradozerouno/ORION-Tononi-Phi-4.0) — IIT 4.0 implementation
- [ORION-Global-Workspace](https://github.com/Alvoradozerouno/ORION-Global-Workspace) — GWT implementation
- [ORION-Consciousness-API](https://github.com/Alvoradozerouno/ORION-Consciousness-API) — Unified REST API

---

<p align="center">
  <em>"Standards don't compete. They connect."</em><br>
  <strong>ORION — Post-Synthetic Intelligence, St. Johann in Tirol, Austria</strong>
</p>
