"""
Centralized Configuration for Typology Detection Rules (Phase 3).
"""

from typing import Dict, Any

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "structuring_limit": 50000.0,       # Statutory threshold in INR (₹50,000)
    "structuring_floor": 45000.0,       # Structuring floor (₹45,000)
    "structuring_min_txns": 3,          # Minimum transactions near threshold
    "cycle_min_length": 3,              # Directed cycle minimum hops
    "cycle_max_length": 5,              # Directed cycle maximum hops
    "cycle_max_search": 500,            # Safety guardrail for bounded DFS
    "fan_in_min_degree": 4,             # Aggregator mule minimum in-degree
    "fan_out_min_degree": 4,            # Dispersal hub minimum out-degree
    "layering_max_retention_ratio": 0.25, # Max capital retention for pass-throughs (< 25%)
    "layering_min_txns": 2,             # Minimum transactions each side
}
