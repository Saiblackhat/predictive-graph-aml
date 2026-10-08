"""
evaluate.py
Macro-averaged F_0.5 scoring, exactly as defined in the problem statement:
  - computed per Source-1 entity, then averaged
  - a correctly-predicted empty list (true singleton) scores 1.0
  - any predicted match on a true singleton scores 0.0
"""
from typing import Dict, Set


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f_beta(pred: Dict[str, Set[str]], gold: Dict[str, Set[str]], beta: float = 0.5) -> float:
    """
    pred, gold: {source1_entity_id: set(matched_ids)}
    Every key in gold must be scored (missing keys treated as empty prediction).
    """
    scores = []
    for sid, gold_set in gold.items():
        pred_set = pred.get(sid, set())
        if len(gold_set) == 0 and len(pred_set) == 0:
            scores.append(1.0)
            continue
        if len(pred_set) == 0:
            scores.append(0.0)  # recall = 0 -> F = 0 (unless gold also empty, handled above)
            continue
        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set) if len(gold_set) else 0.0
        scores.append(f_beta(precision, recall, beta))
    return sum(scores) / len(scores) if scores else 0.0


def parse_id_list(s) -> Set[str]:
    if s is None or (isinstance(s, float)):  # NaN
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return set(x.strip() for x in s.split(",") if x.strip())
