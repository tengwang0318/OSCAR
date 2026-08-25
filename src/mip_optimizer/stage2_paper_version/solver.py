import time
from typing import List, Dict, Set, Tuple
import numpy as np
import coptpy as cp
from coptpy import COPT


class TwoClauseSolver:
    """Two-Clause MIP Solver: Fixed structure for stable trajectories.
    
    Decision Variables:
    - x_j: binary, whether positive tool j is selected
    - w_k: binary, whether negative tool k is selected
    - p_u: binary, whether item u is in positive union
    - g_u: binary, whether item u is in negative intersect
    - r_u: binary, whether item u is in final result
    - n_used: binary, whether negative clause is used
    
    Constraints:
    - Positive tools: min_positive_tools ≤ ∑x_j ≤ max_positive_tools
    - Positive union: p_u = OR(x_j for tools containing u)
    - Negative clause usage: n_used = 1 iff ∑w_k ≥ 1
    - Negative intersect: g_u = AND(w_k for tools containing u) if n_used=1, else 0
    - Final result: r_u = p_u AND NOT g_u
    - Negative tools: min_negative_tools · n_used ≤ ∑w_k ≤ max_negative_tools
    
    Objective: max ∑ h_u · r_u + diversity_bonus + tool_penalty
    """
    
    def __init__(self, 
                 positive_tools: List[Dict],
                 negative_tools: List[Dict],
                 tool_results: Dict[str, List[str]],
                 negative_retrieved: Dict[str, List[str]],
                 llm_scores: Dict[str, float],
                 positive_universe: List[str],
                 ground_truth: List[str],
                 score_threshold: float,
                 min_positive_tools: int,
                 max_positive_tools: int,
                 min_negative_tools: int,
                 max_negative_tools: int,
                 diversity_weight: float,
                 tool_penalty: float,
                 negation_reward: float,
                 topk_weight: float,
                 precision_weight: float,
                 time_limit: float = None,
                 verbose: bool = True):
        """Initialize Two-Clause MIP Solver.
        
        Args:
            positive_tools: List of positive tool dicts (is_negation=False)
            negative_tools: List of negative tool dicts (is_negation=True)
            tool_results: Dict mapping tool_id -> retrieved image list (for positive tools)
            negative_retrieved: Dict mapping negative tool_id -> FULL B_k (complete retrieval results)
            llm_scores: Dict mapping image filename -> relevance score [0,1]
            positive_universe: List of candidate images (U+)
            ground_truth: List of ground truth images (for evaluation)
            score_threshold: Threshold for h_u (default: 0.8)
            min_positive_tools: Minimum positive tools (default: 3)
            max_positive_tools: Maximum positive tools (default: 8)
            min_negative_tools: Minimum negative tools (when negative clause is used, default: 2)
            max_negative_tools: Maximum negative tools (default: 6)
            diversity_weight: Bonus for using diverse tool types
            tool_penalty: Penalty for using more tools
            negation_reward: Reward for using negative clause (encourages negation when beneficial)
            topk_weight: Weight for top-k bonus (encourages larger retrieval scope for ranker)
            precision_weight: Weight for precision bonus (encourages high-quality tools)
            time_limit: Solver time limit in seconds
            verbose: Print solver progress
        """
        self.positive_tools = positive_tools
        self.negative_tools = negative_tools
        self.tool_results = tool_results
        self.negative_retrieved = negative_retrieved
        self.llm_scores = llm_scores
        self.positive_universe = positive_universe
        self.ground_truth = set(ground_truth)
        self.score_threshold = score_threshold
        self.min_positive_tools = min_positive_tools
        self.max_positive_tools = max_positive_tools
        self.min_negative_tools = min_negative_tools
        self.max_negative_tools = max_negative_tools
        self.diversity_weight = diversity_weight
        self.tool_penalty = tool_penalty
        self.negation_reward = negation_reward
        self.topk_weight = topk_weight
        self.precision_weight = precision_weight
        self.time_limit = time_limit
        self.verbose = verbose
        
        # Build indices
        self.u_plus_set = set(positive_universe)
        self.u_idx = {img: i for i, img in enumerate(positive_universe)}
        self.U = len(positive_universe)
        self.J_plus = len(positive_tools)
        self.J_minus = len(negative_tools)
        
        # Precompute high-confidence indicators
        self.h = {img: 1.0 if llm_scores.get(img, 0) >= score_threshold else 0.0
                  for img in positive_universe}
        
        # Build membership matrices
        self._build_membership_matrices()
        
        # Build tool type diversity mapping
        self._build_tool_type_mapping()
        
        # Precompute tool quality metrics (for objective bonuses)
        self._compute_tool_quality_metrics()
    
    def _build_membership_matrices(self):
        """Build binary matrices for tool membership.
        
        a[u, j] = 1 if item u ∈ S_j (positive tool j)
        b[u, k] = 1 if item u ∈ B_k (negative tool k's ORIGINAL results, NOT complement!)
        
        Important: B_k is the original top-m results of negative tool k (negative evidence).
                   Negative intersect = ∩B_k = items retrieved by ALL selected negative tools.
                   This represents "strong negative evidence".
        """
        self.a = np.zeros((self.U, self.J_plus), dtype=int)
        self.b = np.zeros((self.U, self.J_minus), dtype=int)
        
        # Positive tools
        for j, tool in enumerate(self.positive_tools):
            tool_id = tool["id"]
            if tool_id in self.tool_results:
                for img in self.tool_results[tool_id]:
                    if img in self.u_idx:
                        self.a[self.u_idx[img], j] = 1
        
        # Negative tools: use negative_retrieved (full B_k)
        # negative_retrieved stores the COMPLETE retrieval results for each negative tool
        # We filter by U+ here to get B_k ∩ U+ (negative evidence within candidates)
        # b[u, k] = 1 means item u was retrieved by negative tool k
        for k, tool in enumerate(self.negative_tools):
            tool_id = tool["id"]
            if tool_id in self.negative_retrieved:
                for img in self.negative_retrieved[tool_id]:  # Full B_k (e.g., 50 items)
                    if img in self.u_idx:  # Filter by U+ to get B_k ∩ U+
                        self.b[self.u_idx[img], k] = 1
    
    def _build_tool_type_mapping(self):
        """Build mapping for tool type diversity bonus."""
        all_tool_names = set()
        for tool in self.positive_tools + self.negative_tools:
            all_tool_names.add(tool["name"])
        
        self.tool_types = {name: idx for idx, name in enumerate(sorted(all_tool_names))}
        self.num_types = len(self.tool_types)
        
        # For each tool, get its type index
        self.pos_tool_types = [self.tool_types[t["name"]] for t in self.positive_tools]
        self.neg_tool_types = [self.tool_types[t["name"]] for t in self.negative_tools]
    
    def _compute_tool_quality_metrics(self):
        """Precompute quality metrics for each positive tool.
        
        For objective bonus terms (ONLY for positive tools):
        - topk_normalized: retrieval scope normalized to [0, 1]
        - precision: high-score ratio (already in [0, 1])
        
        Rationale: Since we have a powerful VLM ranker downstream, we want positive tools to:
        1. Provide sufficient candidates (large topk) for high recall
        2. Maintain quality to reduce ranker burden (high precision)
        
        Note: Negative tools are NOT rewarded for topk/precision, as their role is filtering, not recall.
        """
        # Reference values for normalization
        self.max_topk = 50.0  # Typical maximum topk in our tool set
        
        self.pos_tool_quality = []
        
        for j, tool in enumerate(self.positive_tools):
            topk = tool.get("top_k", 50)
            
            # Normalize topk to [0, 1]
            topk_normalized = min(topk / self.max_topk, 1.0)
            
            # Calculate precision: ratio of high-score images in this tool's retrieval
            tool_id = tool["id"]
            if tool_id in self.tool_results:
                retrieved = self.tool_results[tool_id]
                high_score_count = sum(1 for img in retrieved if self.h.get(img, 0) > 0.5)
                precision = high_score_count / topk if topk > 0 else 0.0
            else:
                precision = 0.0
            
            # Composite quality score (in [0, 1])
            # This will be weighted by topk_weight and precision_weight in the objective
            quality = {
                'topk_norm': topk_normalized,
                'precision': precision
            }
            self.pos_tool_quality.append(quality)
    
    def solve(self) -> Dict:
        """Solve the Two-Clause MIP.
        
        Returns:
            Dict with solution info
        """
        if self.verbose:
            print(f"📊 Two-Clause Solver: |U+|={self.U}, |J+|={self.J_plus}, |J-|={self.J_minus}")
            print(f"   Threshold: {self.score_threshold}, High-conf items: {sum(self.h.values()):.0f}")
            print(f"   Constraints: pos=[{self.min_positive_tools},{self.max_positive_tools}], neg=[{self.min_negative_tools},{self.max_negative_tools}]")
        
        start = time.time()
        
        # Build model
        model = cp.Envr().createModel("two_clause_v6")
        model.setParam(COPT.Param.Logging, 0)
        if self.time_limit:
            model.setParam(COPT.Param.TimeLimit, self.time_limit)
        
        # ========== Decision Variables ==========
        # Tool selection
        x = model.addVars(self.J_plus, vtype=COPT.BINARY, nameprefix="x")  # Positive tools
        w = model.addVars(self.J_minus, vtype=COPT.BINARY, nameprefix="w")  # Negative tools
        
        # Item membership
        p = model.addVars(self.U, vtype=COPT.BINARY, nameprefix="p")  # In positive union
        g = model.addVars(self.U, vtype=COPT.BINARY, nameprefix="g")  # In negative intersect
        r = model.addVars(self.U, vtype=COPT.BINARY, nameprefix="r")  # In final result
        
        # Negative clause usage indicator
        n_used = model.addVar(vtype=COPT.BINARY, name="n_used")
        
        # Tool type diversity (if enabled)
        if self.diversity_weight > 0:
            type_used = model.addVars(self.num_types, vtype=COPT.BINARY, nameprefix="type")
        
        # ========== Constraints ==========
        
        # 0. Positive tools constraints: min_positive <= sum(x) <= max_positive
        pos_sum = cp.quicksum(x[j] for j in range(self.J_plus))
        model.addConstr(pos_sum >= self.min_positive_tools, name="pos_tool_min")
        model.addConstr(pos_sum <= self.max_positive_tools, name="pos_tool_max")
        
        # 1. Positive Clause = Union (OR-linearization)
        for u in range(self.U):
            # p[u] >= a[u,j] * x[j] for all j
            for j in range(self.J_plus):
                if self.a[u, j] == 1:
                    model.addConstr(p[u] >= x[j], name=f"pos_union_ub_{u}_{j}")
            
            # p[u] <= sum(a[u,j] * x[j])
            model.addConstr(
                p[u] <= cp.quicksum(self.a[u, j] * x[j] for j in range(self.J_plus)),
                name=f"pos_union_lb_{u}"
            )
        
        # 2. Negative Clause Usage Indicator
        if self.J_minus > 0:
            # n_used = 1 iff sum(w[k]) >= 1
            model.addConstr(
                cp.quicksum(w[k] for k in range(self.J_minus)) >= n_used,
                name="neg_usage_lb"
            )
            model.addConstr(
                cp.quicksum(w[k] for k in range(self.J_minus)) <= self.J_minus * n_used,
                name="neg_usage_ub"
            )
        else:
            model.addConstr(n_used == 0, name="no_neg_tools")
        
        # 3. Negative Clause = Intersect (AND-linearization)
        if self.J_minus > 0:
            for u in range(self.U):
                # Upper bounds: g[u] <= b[u,k] + (1 - w[k]) for all k
                for k in range(self.J_minus):
                    model.addConstr(
                        g[u] <= self.b[u, k] + (1 - w[k]),
                        name=f"neg_intersect_ub_{u}_{k}"
                    )
                
                # Lower bound: g[u] >= n_used - sum((1 - b[u,k]) * w[k])
                model.addConstr(
                    g[u] >= n_used - cp.quicksum((1 - self.b[u, k]) * w[k] for k in range(self.J_minus)),
                    name=f"neg_intersect_lb_{u}"
                )
                
                # Disable when negative clause is empty
                model.addConstr(g[u] <= n_used, name=f"neg_disable_{u}")
        else:
            for u in range(self.U):
                model.addConstr(g[u] == 0, name=f"no_neg_{u}")
        
        # 4. Final Result = Difference (p AND NOT g)
        for u in range(self.U):
            model.addConstr(r[u] <= p[u], name=f"result_in_pos_{u}")
            model.addConstr(r[u] <= 1 - g[u], name=f"result_not_neg_{u}")
            model.addConstr(r[u] >= p[u] - g[u], name=f"result_lb_{u}")
        
        # 5. Negative tools constraints: min_negative * n_used <= sum(w) <= max_negative
        if self.J_minus > 0:
            neg_sum = cp.quicksum(w[k] for k in range(self.J_minus))
            
            # Upper bound: at most max_negative_tools
            model.addConstr(neg_sum <= self.max_negative_tools, name="neg_tool_max")
            
            # Lower bound: when negative clause is used (n_used=1), must select at least min_negative_tools
            # This ensures: if n_used=1, then sum(w) >= min_negative_tools
            model.addConstr(neg_sum >= self.min_negative_tools * n_used, name="neg_tool_min")
        
        # 6. Tool Type Diversity (if enabled)
        diversity_term = 0
        if self.diversity_weight > 0:
            # For each tool type, track if it's used
            for type_name, type_idx in self.tool_types.items():
                # Find positive tools of this type
                pos_tools_of_type = [j for j in range(self.J_plus) 
                                    if self.pos_tool_types[j] == type_idx]
                # Find negative tools of this type
                neg_tools_of_type = [k for k in range(self.J_minus) 
                                    if self.neg_tool_types[k] == type_idx]
                
                if pos_tools_of_type or neg_tools_of_type:
                    # type_used[type_idx] = 1 iff at least one tool of this type is selected
                    total_tools = (
                        cp.quicksum(x[j] for j in pos_tools_of_type) +
                        cp.quicksum(w[k] for k in neg_tools_of_type)
                    )
                    model.addConstr(type_used[type_idx] <= total_tools)
                    model.addConstr(total_tools <= (len(pos_tools_of_type) + len(neg_tools_of_type)) * type_used[type_idx])
            
            diversity_term = self.diversity_weight * cp.quicksum(type_used[t] for t in range(self.num_types))
        
        # ========== Objective ==========
        # Maximize: sum(h[u] * r[u]) + bonuses - penalty
        # Rationale: With a powerful VLM ranker downstream, we want to:
        # 1. Cover high-confidence items (primary objective)
        # 2. Provide sufficient candidate pool for ranker (topk bonus)
        # 3. Maintain quality to reduce ranker burden (precision bonus)
        
        high_conf_term = cp.quicksum(self.h[self.positive_universe[u]] * r[u] for u in range(self.U))
        
        tool_count = (
            cp.quicksum(x[j] for j in range(self.J_plus)) +
            cp.quicksum(w[k] for k in range(self.J_minus))
        )
        penalty_term = self.tool_penalty * tool_count
        
        # Negation reward: bonus for using negative clause (n_used = 1)
        negation_term = self.negation_reward * n_used
        
        # Tool quality bonuses (ONLY for positive tools)
        # Rationale: 
        # - Positive tools need large topk for recall (downstream ranker will handle precision)
        # - Negative tools are for filtering, don't need large topk
        # - Use normalized scores to avoid "more tools = more reward" problem
        #
        # Each tool contributes a quality score in [0, topk_weight + precision_weight]
        # Typical per-tool contribution: 0.0001 + 0.01 = 0.0101
        # For 5 tools: ~0.05, which is small compared to primary objective (5-50)
        
        quality_term = 0
        if self.topk_weight > 0 or self.precision_weight > 0:
            for j in range(self.J_plus):
                # Each positive tool's quality contribution
                tool_quality = (
                    self.topk_weight * self.pos_tool_quality[j]['topk_norm'] +
                    self.precision_weight * self.pos_tool_quality[j]['precision']
                )
                quality_term += x[j] * tool_quality
        
        objective = high_conf_term + diversity_term + negation_term + quality_term - penalty_term
        model.setObjective(objective, COPT.MAXIMIZE)
        
        # ========== Solve ==========
        model.solve()
        
        solve_time = time.time() - start
        
        if model.status != COPT.OPTIMAL:
            status_map = {COPT.TIMEOUT: "timeout", COPT.INFEASIBLE: "infeasible"}
            status = status_map.get(model.status, f"status_{model.status}")
            if self.verbose:
                print(f"⚠️  Solver status: {status}")
            return {
                "selected_positive_tools": [],
                "selected_negative_tools": [],
                "positive_union": [],
                "negative_intersect": [],
                "final_result": [],
                "high_confidence_covered": 0,
                "status": status,
                "solve_time": solve_time,
                "objective_value": 0.0
            }
        
        # ========== Extract Solution ==========
        selected_pos_indices = [j for j in range(self.J_plus) if x[j].x > 0.5]
        selected_neg_indices = [k for k in range(self.J_minus) if w[k].x > 0.5]
        
        positive_union = [self.positive_universe[u] for u in range(self.U) if p[u].x > 0.5]
        negative_intersect = [self.positive_universe[u] for u in range(self.U) if g[u].x > 0.5]
        final_result = [self.positive_universe[u] for u in range(self.U) if r[u].x > 0.5]
        
        # Count high-confidence items covered
        high_conf_covered = sum(1 for img in final_result if self.h.get(img, 0) > 0.5)
        
        if self.verbose:
            print(f"✅ Solved in {solve_time:.2f}s")
            print(f"   Selected: {len(selected_pos_indices)} positive + {len(selected_neg_indices)} negative tools")
            print(f"   Result: |P|={len(positive_union)}, |N|={len(negative_intersect)}, |R|={len(final_result)}")
            print(f"   High-conf covered: {high_conf_covered}/{sum(self.h.values()):.0f}")
        
        return {
            "selected_positive_tools": [self.positive_tools[j] for j in selected_pos_indices],
            "selected_negative_tools": [self.negative_tools[k] for k in selected_neg_indices],
            "positive_union": positive_union,
            "negative_intersect": negative_intersect,
            "final_result": final_result,
            "high_confidence_covered": high_conf_covered,
            "status": "optimal",
            "solve_time": solve_time,
            "objective_value": float(model.objval)
        }
