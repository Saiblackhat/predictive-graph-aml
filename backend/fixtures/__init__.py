"""Synthetic hospital discharge fixtures for CAREBRIDGE safety testing."""

from backend.fixtures.discharge_cases import (
    ALL_SYNTHETIC_CASES,
    build_ambiguous_case,
    build_conflicting_case,
    build_incomplete_case,
    build_normal_case,
    build_unsafe_case,
)

__all__ = [
    "ALL_SYNTHETIC_CASES",
    "build_ambiguous_case",
    "build_conflicting_case",
    "build_incomplete_case",
    "build_normal_case",
    "build_unsafe_case",
]
