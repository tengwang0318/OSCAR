"""Generate candidate clauses and precompute coverage matrices."""

from typing import List, Dict, Tuple
import numpy as np
from itertools import combinations
from .schemas import Clause


def generate_clauses_and_matrices(selected_tools, all_tools, tool_results, u_plus, ground_truth,
                                  max_clause_len, negation_complements,
                                  max_negation_per_clause):
    """
    Generate candidate clauses: I_c = ∩ S_p (intersection of positive tools only)

    Rules:
    - |P_c| >= 1 (at least 1 positive literal from S')
    - |P_c| <= max_clause_len
    - At most max_negation_per_clause negation tools per clause (default: 1)

    Note: Difference operations (A - B) are converted to intersection (A ∩ -B) in Stage 1.
          Negation tool results are replaced with their complements (U+ - B).
          Stage 2 only performs union and intersection operations.

    Args:
        negation_complements: Dict mapping negation tool IDs to their complement results (U+ - B)
        max_negation_per_clause: Maximum number of negation tools allowed per clause (default: 1)
    """
    clauses = []
    u_plus_set = set(u_plus)
    non_gt = sorted(u_plus_set - set(ground_truth))

    if negation_complements is None:
        negation_complements = {}

    # Build mappings: use complement results for negation tools
    tool_res = {}
    for t in all_tools:
        if t["id"] not in tool_results:
            continue

        # Use complement if this is a negation tool
        if t["id"] in negation_complements:
            tool_res[t["id"]] = set(negation_complements[t["id"]])
        else:
            tool_res[t["id"]] = set(tool_results[t["id"]])

    # Build tool_map with retrieved results and is_negation field
    tool_map = {}
    for t in all_tools:
        tool_info = t.copy()
        if t["id"] in tool_results:
            # Keep original order (ranked by similarity), don't sort!
            tool_info["retrieved_images"] = list(tool_results[t["id"]])
        # Ensure is_negation field is present (default to False if not specified)
        if "is_negation" not in tool_info:
            tool_info["is_negation"] = False
        tool_map[t["id"]] = tool_info

    sel_ids = [t["id"] for t in selected_tools]
    all_ids = [t["id"] for t in all_tools]

    # Generate clauses: I_c = S_p1 ∩ S_p2 ∩ ... (intersection only)
    for num_pos in range(1, min(max_clause_len + 1, len(sel_ids) + 1)):
        for pos_ids in combinations(sel_ids, num_pos):
            # Check all have results
            if not all(tid in tool_res for tid in pos_ids):
                continue

            # Check negation constraint: at most max_negation_per_clause negation tools
            negation_count = sum(1 for tid in pos_ids if tool_map[tid].get("is_negation", False))
            if negation_count > max_negation_per_clause:
                continue

            # Compute: (S_p1 ∩ S_p2 ∩ ...) ∩ U+
            result = u_plus_set.copy()
            for tid in pos_ids:
                result &= tool_res[tid]

            if result:
                # Separate positive and negative tools for cleaner output
                pos_tools = [tool_map[tid] for tid in pos_ids if not tool_map[tid].get("is_negation", False)]
                neg_tools = [tool_map[tid] for tid in pos_ids if tool_map[tid].get("is_negation", False)]
                
                # Skip if no positive tools - clause must have at least 1 positive tool as base
                # (negation/DIFFERENCE needs a base set to subtract from)
                if not pos_tools:
                    continue
                
                clauses.append(Clause(
                    type="single" if len(pos_tools) == 1 else "intersection",
                    positive_tools=pos_tools,
                    negative_tools=neg_tools,  # At most 1 negation tool per clause
                    result_set=sorted(result)
                ))

    # Precompute A and B matrices
    A = np.zeros((len(ground_truth), len(clauses)), dtype=int)
    B = np.zeros((len(non_gt), len(clauses)), dtype=int)

    gt_idx = {img: i for i, img in enumerate(ground_truth)}
    non_gt_idx = {img: i for i, img in enumerate(non_gt)}

    for c, clause in enumerate(clauses):
        for img in clause.result_set:
            if img in gt_idx:
                A[gt_idx[img], c] = 1
            elif img in non_gt_idx:
                B[non_gt_idx[img], c] = 1

    return clauses, A, B

