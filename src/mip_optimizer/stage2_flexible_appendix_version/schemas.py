"""Data schemas for Stage 2 solutions."""

from typing import List, Dict, Any
from pydantic import BaseModel


class Clause(BaseModel):
    """A candidate clause for logical composition.

    Clause represents: (∩ positive_tools) ∩ (∩ complement(negative_tools))
    
    Note: negative_tools' results are already converted to complement (U+ - B) in Stage 1.
          At most 1 negation tool per clause (controlled by max_negation_per_clause).
    """
    type: str  # "single" | "intersection"
    positive_tools: List[Dict[str, Any]]  # Tools with is_negation=False
    negative_tools: List[Dict[str, Any]] = []  # Tools with is_negation=True (at most 1)
    result_set: List[str]


class Stage2Solution(BaseModel):
    """Stage 2 solution: logical composition."""
    query_id: str
    query_text: str
    ground_truth: List[str]  # Ground truth images for evaluation
    selected_clauses: List[Clause]
    final_result: List[str]
    f1: float
    beta: float  # F-beta parameter (beta=1 for traditional F1)
    dinkelbach_iterations: int
    solve_time: float

