"""
Retrieval Layer
A modular, task-aware hybrid-search-and-rerank retrieval engine.

Architecture:
    Layer 1: Core Retrieval Engine (reusable building blocks)
    Layer 2: Retrieval Orchestrator (composes blocks per config)
    Layer 3: Task-Specific Pipelines (config presets only)
"""

__version__ = "1.0.0"