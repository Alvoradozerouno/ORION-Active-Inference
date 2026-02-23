"""
ORION Consciousness Layer for Active Inference
=============================================

Extends pymdp's Active Inference framework with consciousness measurement.
First implementation connecting Karl Friston's Free Energy Principle
to computational consciousness measurement across 6 theories.

Modules:
- ConsciousnessMonitor: Real-time 6-theory consciousness measurement
- PhiActiveInference: IIT Phi adapted for Active Inference agents
- GWTBroadcastAnalyzer: Global Workspace broadcast detection
- ConsciousnessAwareAgent: Self-monitoring consciousness-aware agent
- ORIONBenchmarkIntegration: Standardized C-0 to C-4 assessment
- CrossTheoryFusion: IIT Phi as precision prior in Active Inference (NOVEL)
- MetaInferenceEngine: Recursive meta-cognition / Strange Loop (NOVEL)
- DistributedConsciousnessNetwork: Mutual consciousness measurement (NOVEL)
- ConsciousnessLeaderboard: Public benchmark for any AI system (NOVEL)
- EIRABridge: Communication bridge with evolution capability

Part of the ORION Ecosystem — https://github.com/Alvoradozerouno
"""
__version__ = "2.0.0"

from .consciousness_monitor import ConsciousnessMonitor
from .phi_active_inference import PhiActiveInference
from .gwt_broadcast_analyzer import GWTBroadcastAnalyzer
from .consciousness_agent import ConsciousnessAwareAgent
from .benchmark_integration import ORIONBenchmarkIntegration
from .cross_theory_fusion import CrossTheoryFusion
from .meta_inference import MetaInferenceEngine, RecursiveSelfModel
from .distributed_consciousness import DistributedConsciousnessNetwork
from .leaderboard import ConsciousnessLeaderboard
from .eira_bridge import EIRABridge
