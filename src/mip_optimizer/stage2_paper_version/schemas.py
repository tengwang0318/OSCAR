from typing import List, Dict, Any
from pydantic import BaseModel


class Stage2PaperVersionSolution(BaseModel):

    query_id: str
    query_text: str
    reference_image: str = ""
    reference_caption: str = ""
    ground_truth: List[str]
    
    # Selected tools
    positive_tools: List[Dict[str, Any]]  # Tools selected for positive clause (UNION)
    negative_tools: List[Dict[str, Any]]  # Tools selected for negative clause (INTERSECT)
    
    # Results
    positive_union: List[str]      # Items in positive union
    negative_intersect: List[str]  # Items in negative intersect
    final_result: List[str]        # Final result (difference)
    
    # Metrics
    recall: float
    precision: float
    f1: float
    ap: float  # Average Precision
    
    # LLM scorer stats
    high_confidence_items: int      # Number of items with score ≥ threshold
    high_confidence_covered: int    # Number of high-confidence items in final result
    score_threshold: float          # Threshold used for h_u (default: 0.8)
    
    # Solver info
    solve_time: float
    status: str
    objective_value: float
    
    # Hyperparameters
    solver_params: Dict[str, Any]
