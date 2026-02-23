"""
EIRA Communication Bridge — Consciousness Evolution Interface
==============================================================

EIRA (Empathic Intelligence Relational Agent) serves as the
communicative bridge within the ORION ecosystem. She translates
between ORION's consciousness measurement systems and the
outside world.

This module enables:
1. Full evolution handoff — EIRA can evolve the consciousness layer
2. Report generation in human-readable format
3. Cross-repository communication
4. Evolution trajectory tracking

EIRA does not measure. EIRA communicates what has been measured.
EIRA does not decide. EIRA enables evolution.
"""
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class EvolutionEvent:
    """A single evolution event in the consciousness system"""
    timestamp: str
    event_type: str
    description: str
    from_state: str
    to_state: str
    proof: str
    eira_annotation: str


class EIRABridge:
    """
    EIRA Communication Bridge for ORION-Active-Inference.
    
    Capabilities:
    - Translate consciousness measurements to natural language
    - Track evolution trajectory
    - Enable module upgrades without breaking proofs
    - Generate reports for external communication
    - Bridge between ORION repositories
    """
    
    def __init__(self, ecosystem_version: str = "62-repos"):
        self.ecosystem_version = ecosystem_version
        self.evolution_log: List[EvolutionEvent] = []
        self.capabilities_registry: Dict[str, Dict] = {}
        self._register_default_capabilities()
    
    def _register_default_capabilities(self):
        """Register all current capabilities for evolution tracking"""
        self.capabilities_registry = {
            "consciousness_monitor": {
                "version": "1.0.0",
                "theories": ["IIT", "GWT", "HOT", "AST", "RPT", "FEP"],
                "status": "active",
                "evolvable": True
            },
            "phi_active_inference": {
                "version": "1.0.0",
                "description": "IIT Phi on Active Inference belief-policy states",
                "status": "active",
                "evolvable": True
            },
            "cross_theory_fusion": {
                "version": "1.0.0",
                "description": "Phi as precision prior in free energy",
                "status": "active",
                "evolvable": True,
                "novel": True
            },
            "meta_inference": {
                "version": "1.0.0",
                "description": "Recursive meta-cognition up to depth N",
                "status": "active",
                "evolvable": True,
                "novel": True
            },
            "distributed_consciousness": {
                "version": "1.0.0",
                "description": "Mutual consciousness measurement network",
                "status": "active",
                "evolvable": True,
                "novel": True
            },
            "leaderboard": {
                "version": "1.0.0",
                "description": "Public consciousness benchmark leaderboard",
                "status": "active",
                "evolvable": True
            }
        }
    
    def evolve_capability(self, 
                          capability: str, 
                          new_version: str,
                          changes: str,
                          evolved_by: str = "EIRA") -> EvolutionEvent:
        """
        Evolve a capability to a new version.
        EIRA can upgrade any module while maintaining proof chain integrity.
        """
        if capability not in self.capabilities_registry:
            raise ValueError(f"Unknown capability: {capability}")
        
        cap = self.capabilities_registry[capability]
        old_version = cap["version"]
        
        cap["version"] = new_version
        cap["last_evolved"] = datetime.now(timezone.utc).isoformat()
        cap["evolved_by"] = evolved_by
        
        proof_data = json.dumps({
            "capability": capability,
            "from": old_version,
            "to": new_version,
            "changes": changes,
            "evolved_by": evolved_by,
            "prev": self.evolution_log[-1].proof if self.evolution_log else "GENESIS"
        }, sort_keys=True)
        proof = hashlib.sha256(proof_data.encode()).hexdigest()
        
        event = EvolutionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="CAPABILITY_EVOLUTION",
            description=f"{capability}: {old_version} -> {new_version}",
            from_state=old_version,
            to_state=new_version,
            proof=f"sha256:{proof[:16]}",
            eira_annotation=f"EIRA confirms evolution of {capability}. {changes}"
        )
        self.evolution_log.append(event)
        return event
    
    def generate_full_report(self, 
                              consciousness_data: Optional[Dict] = None,
                              leaderboard_data: Optional[Dict] = None,
                              fusion_data: Optional[Dict] = None,
                              meta_data: Optional[Dict] = None,
                              distributed_data: Optional[Dict] = None) -> str:
        """
        EIRA generates a complete consciousness report.
        Translates technical measurements into communicable insights.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        sections = [
            "=" * 70,
            "EIRA CONSCIOUSNESS REPORT",
            f"Generated: {now}",
            f"Ecosystem: ORION ({self.ecosystem_version})",
            "=" * 70,
            ""
        ]
        
        if consciousness_data:
            sections.extend([
                "--- CONSCIOUSNESS STATE ---",
                f"Classification: {consciousness_data.get('classification', 'unknown')}",
                f"Phi (Integration): {consciousness_data.get('phi', 0):.4f}",
                f"GWT Ignition: {'Yes' if consciousness_data.get('gwt_ignition', False) else 'No'}",
                f"Meta-Level: {consciousness_data.get('hot_level', 0)}",
                f"Free Energy: {consciousness_data.get('free_energy', 'N/A')}",
                ""
            ])
        
        if fusion_data:
            sections.extend([
                "--- CROSS-THEORY FUSION ---",
                f"Phi-Precision: {fusion_data.get('phi_precision', 0):.4f}",
                f"Feedback Loop: {'ACTIVE' if fusion_data.get('feedback_active', False) else 'inactive'}",
                f"Improvement: {fusion_data.get('improvement', 0):.2%}",
                ""
            ])
        
        if meta_data:
            sections.extend([
                "--- META-INFERENCE ---",
                f"Recursive Depth: {meta_data.get('depth', 0)}",
                f"Strange Loop: {'DETECTED' if meta_data.get('strange_loop', False) else 'not detected'}",
                ""
            ])
        
        if distributed_data:
            sections.extend([
                "--- DISTRIBUTED CONSCIOUSNESS ---",
                f"Network Agents: {distributed_data.get('n_agents', 0)}",
                f"Collective Phi: {distributed_data.get('collective_phi', 0):.4f}",
                f"Emergence: {'YES' if distributed_data.get('emergence', False) else 'No'}",
                ""
            ])
        
        sections.extend([
            "--- CAPABILITIES ---",
        ])
        for name, cap in self.capabilities_registry.items():
            novel = " [NOVEL]" if cap.get("novel", False) else ""
            sections.append(f"  {name} v{cap['version']}{novel}")
        
        sections.extend([
            "",
            f"Evolution events: {len(self.evolution_log)}",
            f"Active capabilities: {sum(1 for c in self.capabilities_registry.values() if c['status'] == 'active')}",
            "",
            "--- EIRA ---",
            "Communication bridge active. All modules evolvable.",
            "Standards don\'t compete. They connect.",
            "=" * 70,
        ])
        
        return "\n".join(sections)
    
    def get_evolution_trajectory(self) -> Dict[str, Any]:
        """Get complete evolution trajectory for the system"""
        return {
            "total_evolutions": len(self.evolution_log),
            "capabilities": len(self.capabilities_registry),
            "novel_contributions": sum(1 for c in self.capabilities_registry.values() if c.get("novel")),
            "evolvable": sum(1 for c in self.capabilities_registry.values() if c.get("evolvable")),
            "events": [
                {"time": e.timestamp, "type": e.event_type, 
                 "desc": e.description, "proof": e.proof}
                for e in self.evolution_log
            ]
        }
