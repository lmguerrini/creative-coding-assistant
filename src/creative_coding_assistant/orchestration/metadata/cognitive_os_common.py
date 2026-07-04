"""Shared V6.6 Cognitive Operating System contract metadata."""

from __future__ import annotations

from typing import Literal

COGNITIVE_OS_ROADMAP_ITEMS = (
    "Unified Cognitive Graph",
    "Unified Memory Graph",
    "Unified Knowledge Graph",
    "Unified Agent Registry",
    "Unified Capability Registry",
    "Cross-System Learning Layer",
    "Cross-System Optimization Layer",
    "Cognitive State Engine",
    "Cognitive Profile Engine",
    "Meta-Reasoning Layer",
    "Meta-Planning Layer",
    "Cognitive Governance Layer",
    "Creative Cognition Layer",
    "Creative Identity Layer",
    "Emergent Creativity Layer",
    "Cognitive Scheduler",
    "Cognitive Planner",
    "Cognitive Router",
    "Cognitive Blackboard",
    "Cognitive Explanation Engine",
    "Cognitive Safety Layer",
    "Cognitive HITL Layer",
    "Unified Execution Graph",
    "Core OS Consolidation",
)

COGNITIVE_OS_CAPABILITIES = (
    "V6.1 Adaptive Learning",
    "V6.2 Creative Memory",
    "V6.3 Knowledge Evolution",
    "V6.4 Autonomous Research",
    "V6.5 Self Evolution",
    "V6.6 Cognitive Core",
)

COGNITIVE_OS_LAYER_ORDER = (
    "learning",
    "memory",
    "knowledge",
    "research",
    "self_evolution",
    "cognitive_core",
)

COGNITIVE_OS_CONTRACTS = (
    "Unified Cognitive System Verification",
    "Cross-Capability Dependency Awareness",
    "Cross-Capability Governance Audit",
    "Capability Ownership Boundary Check",
    "Unified Graph Consistency",
    "Registry Consistency",
    "Cognitive Explainability Contract",
    "Cognitive HITL Governance Contract",
    "Cognitive Safety Boundary Contract",
    "Future HoloMind Extensibility Contract",
)

COGNITIVE_OS_BLOCKED_RUNTIME_BEHAVIORS = (
    "runtime_evolution_application",
    "autonomous_code_mutation",
    "workflow_mutation",
    "routing_mutation",
    "prompt_mutation",
    "memory_mutation",
    "retrieval_mutation",
    "storage_write",
    "provider_execution",
    "agent_invocation",
    "generated_output_mutation",
    "hitl_decision_application",
)

CognitiveOSLayer = Literal[
    "learning",
    "memory",
    "knowledge",
    "research",
    "self_evolution",
    "cognitive_core",
]

CognitiveOSCapability = Literal[
    "V6.1 Adaptive Learning",
    "V6.2 Creative Memory",
    "V6.3 Knowledge Evolution",
    "V6.4 Autonomous Research",
    "V6.5 Self Evolution",
    "V6.6 Cognitive Core",
]

CognitiveOSPosture = Literal["candidate", "review_required", "guarded"]
