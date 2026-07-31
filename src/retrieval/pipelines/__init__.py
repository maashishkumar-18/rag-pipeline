"""
Task-specific retrieval pipeline configurations.
Each pipeline is a preset of PipelineConfig — no custom logic, just configuration.
"""

from .learning import LEARNING_PIPELINE
from .planner import PLANNER_PIPELINE
from .revision import REVISION_PIPELINE
from .qa import QA_PIPELINE
from .quiz import QUIZ_PIPELINE

# Registry of all available pipelines
PIPELINE_REGISTRY = {
    "learning": LEARNING_PIPELINE,
    "planner": PLANNER_PIPELINE,
    "revision": REVISION_PIPELINE,
    "qa": QA_PIPELINE,
    "quiz": QUIZ_PIPELINE,
}

__all__ = [
    "LEARNING_PIPELINE",
    "PLANNER_PIPELINE", 
    "REVISION_PIPELINE",
    "QA_PIPELINE",
    "QUIZ_PIPELINE",
    "PIPELINE_REGISTRY",
]