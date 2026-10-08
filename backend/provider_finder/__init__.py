"""
backend/provider_finder/__init__.py

Phase 7 — Provider Finder public API.
"""
from backend.provider_finder.matcher import find_providers
from backend.provider_finder.types import (
    FreshnessStatus,
    MatchAttribute,
    NeutralityStatus,
    ProviderMatch,
    ProviderSearchQuery,
    ProviderSearchResult,
    ProviderType,
)

__all__ = [
    "find_providers",
    "FreshnessStatus",
    "MatchAttribute",
    "NeutralityStatus",
    "ProviderMatch",
    "ProviderSearchQuery",
    "ProviderSearchResult",
    "ProviderType",
]
