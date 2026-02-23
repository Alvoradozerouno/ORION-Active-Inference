"""
ORION Consciousness Layer for Active Inference
=============================================

Extends pymdp's Active Inference framework with consciousness measurement.
First implementation connecting Karl Friston's Free Energy Principle
to Integrated Information Theory, Global Workspace Theory, and
the ORION Consciousness Benchmark.

Part of the ORION Ecosystem — https://github.com/Alvoradozerouno
"""
__version__ = "1.0.0"

from .consciousness_monitor import ConsciousnessMonitor
from .phi_active_inference import PhiActiveInference
from .gwt_broadcast_analyzer import GWTBroadcastAnalyzer
from .consciousness_agent import ConsciousnessAwareAgent
from .benchmark_integration import ORIONBenchmarkIntegration
