"""Canonical machine-readable values shared by figure tooling."""

APPROVAL_DECISIONS = (
    "pending",
    "user-approved",
    "not-applicable",
)

CAPTION_STATUSES = (
    "not-started",
    "draft",
    "requires-information",
    "verified",
    "user-reviewed",
    "final",
)

IMPLEMENTATION_STATUSES = (
    "planned",
    "legacy",
    "inventoried",
    "configured",
    "structurally-migrated",
    "implemented",
    "draft-rendered",
    "technically-validated",
    "user-reviewed",
    "final",
)

IMPLEMENTED_STATUSES = frozenset(
    {
        "implemented",
        "draft-rendered",
        "technically-validated",
        "user-reviewed",
        "final",
    }
)
RENDERED_STATUSES = frozenset(
    {
        "draft-rendered",
        "technically-validated",
        "user-reviewed",
        "final",
    }
)
FINAL_LIKE_STATUSES = frozenset(
    {
        "technically-validated",
        "user-reviewed",
        "final",
    }
)

PACKAGE_TYPES = (
    "overleaf",
    "complete",
)
PACKAGE_SELECTORS = (
    "all-final-publication",
    "all-final",
)
CAPTION_MODES = (
    "caption-only",
    "complete-environment",
    "disabled",
)

TECHNICALLY_VALIDATED = "technically-validated"
USER_REVIEW_APPROVED = "user-approved"
ELIGIBLE_CAPTION_STATUSES = frozenset({"user-reviewed", "final"})
