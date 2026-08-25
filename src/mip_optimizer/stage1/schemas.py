"""Data schemas for Stage 1 solutions."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Stage1Solution(BaseModel):
    """Stage 1 solution: selected tools and positive universe U+."""
    # Query info
    query_id: str
    query_text: str
    reference_image: str  # Reference image path (for CIR)
    reference_caption: str  # Reference image caption (for CIR)
    ground_truth: List[str]
    
    # Solution
    selected_tools: List[Dict[str, Any]]  # List of tool call dicts
    positive_universe: List[str]  # U+ for Stage 2
    # NO negation_complements - read from phase0_augmented when needed

    # Metrics
    recall: float
    precision: float
    num_tools: int

    # Solver info
    status: str
    solve_time: float
    objective_value: float

