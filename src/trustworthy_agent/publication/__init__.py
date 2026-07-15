"""Shared infrastructure for reproducible ArticleV1 publication figures."""

from trustworthy_agent.publication.figure_data import FigureDataArtifact
from trustworthy_agent.publication.figure_registry import FigureRecord, FigureRegistry
from trustworthy_agent.publication.figure_style import FigureStyle, load_style, publication_style

__all__ = [
    "FigureDataArtifact",
    "FigureRecord",
    "FigureRegistry",
    "FigureStyle",
    "load_style",
    "publication_style",
]
