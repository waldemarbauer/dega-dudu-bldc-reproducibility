"""ArticleV1 canonical-window analysis prerequisites.

This package is deliberately separate from the semantic FSM, model-training,
V6 trend, evidence-bundle, and safety packages.  It prepares deterministic
case-local features and acquisition-level assignment definitions only.
"""

from trustworthy_agent.article_v1.assignments import (
    AssignmentDefinition,
    build_exhaustive_assignments,
    validate_assignment_rows,
)
from trustworthy_agent.article_v1.features import (
    CanonicalWindowInput,
    FeatureExtractionError,
    FeatureExtractionResult,
    WindowFeatureExtractor,
    load_feature_config,
)

__all__ = [
    "AssignmentDefinition",
    "CanonicalWindowInput",
    "FeatureExtractionError",
    "FeatureExtractionResult",
    "WindowFeatureExtractor",
    "build_exhaustive_assignments",
    "load_feature_config",
    "validate_assignment_rows",
]
