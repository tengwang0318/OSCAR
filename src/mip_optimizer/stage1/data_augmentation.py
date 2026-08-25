from typing import List, Dict, Any, Tuple

TOP_K_SPLITS = list(range(5, 54, 5))


def augment_tool_calls(
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, Dict[str, Any]],
        deduplicate: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], List[List[int]]]:
    """
    Augment tool calls by splitting top_k=50 into [5, 10, 15, ..., 50].

    Args:
        tool_calls: Original tool calls (all have top_k=50)
        tool_results: Tool results dict {tc_id: {"retrieved_images": [...]}}
        deduplicate: If True, remove duplicate tools with same (name, top_k, result_set)

    Returns:
        augmented_calls: List of augmented tool calls
        augmented_results: Dict mapping tool_call_id to retrieved images
        families: List of families, where each family is a list of indices
                  of tool calls that are variants of the same tool (only top_k differs)
    """
    augmented_calls = []
    augmented_results = {}
    families = []

    for tc in tool_calls:
        tc_id = tc["id"]
        tool_name = tc["name"]

        if tc_id not in tool_results:
            continue

        full_results = tool_results[tc_id]["retrieved_images"]

        # Extract query and reference_image from args
        args = tc.get("args", {})
        query = args.get("query", "")
        reference_image = args.get("reference_image", "")
        
        # Split into multiple top_k versions - they form a family
        family_indices = []
        for k in TOP_K_SPLITS:
            new_id = f"{tc_id}_{k}"
            new_tc = {
                "id": new_id,
                "name": tool_name,
                "query": query,
                "reference_image": reference_image,
                "top_k": k,
                "is_negation": tc.get("is_negation", False)
            }

            family_indices.append(len(augmented_calls))
            augmented_calls.append(new_tc)
            augmented_results[new_id] = full_results[:k]

        # Record this family
        families.append(family_indices)

    # Deduplicate if requested
    if deduplicate:
        augmented_calls, augmented_results, families = _deduplicate_tools(
            augmented_calls, augmented_results, families
        )

    return augmented_calls, augmented_results, families


def _deduplicate_tools(
        aug_calls: List[Dict[str, Any]],
        aug_results: Dict[str, List[str]],
        families: List[List[int]]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], List[List[int]]]:
    """Remove duplicate tools with same (name, top_k, result_set)."""
    seen = {}  # (name, top_k, frozenset(results)) -> first_index
    keep_indices = []  # Indices to keep
    old_to_new_idx = {}  # Map old index to new index

    for i, tc in enumerate(aug_calls):
        tc_id = tc["id"]
        name = tc["name"]
        top_k = tc.get("top_k", None)
        result_set = frozenset(aug_results[tc_id])
        key = (name, top_k, result_set)

        if key not in seen:
            # First occurrence, keep it
            seen[key] = i
            keep_indices.append(i)
            old_to_new_idx[i] = len(keep_indices) - 1
        else:
            # Duplicate, skip (don't add to keep_indices)
            pass

    # Build deduplicated lists
    dedup_calls = [aug_calls[i] for i in keep_indices]
    dedup_results = {aug_calls[i]["id"]: aug_results[aug_calls[i]["id"]]
                     for i in keep_indices}

    # Update families: remove indices that were deduplicated
    dedup_families = []
    for family in families:
        new_family = [old_to_new_idx[idx] for idx in family if idx in old_to_new_idx]
        if new_family:  # Only keep non-empty families
            dedup_families.append(new_family)

    return dedup_calls, dedup_results, dedup_families
