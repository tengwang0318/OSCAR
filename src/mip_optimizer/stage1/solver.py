"""Stage 1 MIP Solver: Positive Tool Selection using COPT."""

import time
from typing import List, Dict, Any, Optional, Set, Tuple
import coptpy as cp
from coptpy import COPT

from .schemas import Stage1Solution


class Stage1Solver:
    """
    Stage 1 MIP Solver: Select tools to maximize weighted Recall - FPR + Diversity + Universe Size.

    Variables:
        x[j] (tool selected),
        y[i] (GT covered),
        z[k] (FP),
        type_used[t] (tool type used)

    Objective: max w_recall·(Σy[i]/|G|) - w_fpr·(Σz[k]/|N|) + λ·(Σtype_used[t]) + ε·|U+|
        where |U+| = Σy[i] + Σz[k] is the size of the positive universe (union of all selected tools)

    Constraints:
        - Top-k variant exclusivity (at most one variant per family)
        - Tool type diversity (type_used[t] = 1 iff at least one tool of type t is selected)
    """

    def __init__(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: Dict[str, List[str]],
        ground_truth: List[str],
        recall_weight: float,
        fpr_weight: float ,
        diversity_weight: float ,
        universe_size_weight: float ,
        min_tools: Optional[int] = None,
        max_tools: Optional[int] = None,
        families: Optional[List[List[int]]] = None,
        time_limit: Optional[float] = None,
        verbose: bool = True
    ):
        self.tool_calls = tool_calls
        self.tool_results = tool_results
        self.GT = ground_truth
        self.w_recall = recall_weight
        self.w_fpr = fpr_weight
        self.w_diversity = diversity_weight
        self.w_universe = universe_size_weight
        self.min_tools = min_tools
        self.max_tools = max_tools
        self.families = families or []
        self.time_limit = time_limit
        self.verbose = verbose

        # Dimensions
        self.T = len(tool_calls)
        self.G = len(ground_truth)

        # Non-GT images
        all_images = set()
        for results in tool_results.values():
            all_images.update(results)
        self.NonGT = sorted([img for img in all_images if img not in set(ground_truth)])
        self.N = len(self.NonGT)

        # Build indicator matrices
        self.a_matrix = self._build_gt_matrix()
        self.b_matrix = self._build_nongt_matrix()

        # Build tool type mapping for diversity bonus
        self.tool_types, self.tool_type_indices = self._build_tool_type_mapping()

        if self.verbose:
            budget_str = f", K={self.max_tools}" if self.max_tools else ""
            family_str = f", Families={len(self.families)}" if self.families else ""
            diversity_str = f", λ_div={self.w_diversity}" if self.w_diversity > 0 else ""
            print(f"\n📊 Solver: T={self.T}, G={self.G}, N={self.N}, w_recall={self.w_recall}, w_fpr={self.w_fpr}{diversity_str}{budget_str}{family_str}")
            if self.w_diversity > 0 and self.tool_types:
                print(f"   Tool types: {list(self.tool_types.keys())}")
    
    def _build_gt_matrix(self) -> List[List[int]]:
        """Build indicator matrix: a[i][j] = 1 if GT[i] in tool[j]."""
        matrix = []
        for gt_img in self.GT:
            row = []
            for tc in self.tool_calls:
                tc_id = tc["id"]
                results = self.tool_results.get(tc_id, [])
                row.append(1 if gt_img in results else 0)
            matrix.append(row)
        return matrix
    
    def _build_nongt_matrix(self) -> List[List[int]]:
        """Build indicator matrix: b[k][j] = 1 if NonGT[k] in tool[j]."""
        matrix = []
        for nongt_img in self.NonGT:
            row = []
            for tc in self.tool_calls:
                tc_id = tc["id"]
                results = self.tool_results.get(tc_id, [])
                row.append(1 if nongt_img in results else 0)
            matrix.append(row)
        return matrix

    def _build_tool_type_mapping(self) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
        """
        Build mapping from tool types to tool indices.

        Returns:
            tool_types: Dict mapping tool_name -> type_index
            tool_type_indices: Dict mapping tool_name -> list of tool indices
        """
        # Collect unique tool types
        unique_types = sorted(set(tc["name"] for tc in self.tool_calls))
        tool_types = {name: idx for idx, name in enumerate(unique_types)}

        # Map each tool type to its tool indices
        tool_type_indices = {name: [] for name in unique_types}
        for j, tc in enumerate(self.tool_calls):
            tool_name = tc["name"]
            tool_type_indices[tool_name].append(j)

        return tool_types, tool_type_indices
    
    def solve(self) -> Dict[str, Any]:
        """Solve Stage 1 MIP and return solution dict."""
        start_time = time.time()

        # Create model
        env = cp.Envr()
        model = env.createModel("Stage1")
        if self.time_limit:
            model.setParam(COPT.Param.TimeLimit, self.time_limit)
        if not self.verbose:
            model.setParam(COPT.Param.Logging, 0)

        # Variables
        x = model.addVars(self.T, vtype=COPT.BINARY, nameprefix="x")
        y = model.addVars(self.G, vtype=COPT.BINARY, nameprefix="y")
        z = model.addVars(self.N, vtype=COPT.BINARY, nameprefix="z")
        
        # Constraints: GT coverage (OR-linearization)
        for i in range(self.G):
            d_i = sum(self.a_matrix[i])
            if d_i > 0:
                model.addConstr(y[i] <= cp.quicksum(self.a_matrix[i][j] * x[j] for j in range(self.T)))
                model.addConstr(cp.quicksum(self.a_matrix[i][j] * x[j] for j in range(self.T)) <= d_i * y[i])
            else:
                model.addConstr(y[i] == 0)

        # Constraints: FP counting
        for k in range(self.N):
            d_k = sum(self.b_matrix[k])
            if d_k > 0:
                model.addConstr(z[k] <= cp.quicksum(self.b_matrix[k][j] * x[j] for j in range(self.T)))
                model.addConstr(cp.quicksum(self.b_matrix[k][j] * x[j] for j in range(self.T)) <= d_k * z[k])

        # Constraint: Tool budget (optional)
        if self.min_tools is not None:
            model.addConstr(cp.quicksum(x[j] for j in range(self.T)) >= self.min_tools)
            if self.verbose:
                print(f"   Added constraint: Σx[j] >= {self.min_tools}")
        if self.max_tools is not None:
            model.addConstr(cp.quicksum(x[j] for j in range(self.T)) <= self.max_tools)
            if self.verbose:
                print(f"   Added constraint: Σx[j] <= {self.max_tools}")

        # Constraint: Top-k variant exclusivity (at most one variant per family)
        for family in self.families:
            model.addConstr(cp.quicksum(x[j] for j in family) <= 1)
        if self.verbose and self.families:
            print(f"   Added {len(self.families)} family exclusivity constraints")

        # Tool type diversity variables and constraints
        diversity_term = 0
        if self.w_diversity > 0 and self.tool_types:
            # Create binary variables for each tool type
            num_types = len(self.tool_types)
            type_used = model.addVars(num_types, vtype=COPT.BINARY, nameprefix="type_used")

            # Add constraints: type_used[t] = 1 iff at least one tool of type t is selected
            for tool_name, type_idx in self.tool_types.items():
                tools_of_type = self.tool_type_indices[tool_name]

                if tools_of_type:
                    # type_used[t] <= Σ x[j] for j in tools_of_type
                    model.addConstr(
                        type_used[type_idx] <= cp.quicksum(x[j] for j in tools_of_type)
                    )

                    # Σ x[j] <= M * type_used[t], where M = number of tools of this type
                    model.addConstr(
                        cp.quicksum(x[j] for j in tools_of_type) <= len(tools_of_type) * type_used[type_idx]
                    )

            # Diversity bonus: λ * (number of different tool types used)
            diversity_term = self.w_diversity * cp.quicksum(type_used[t] for t in range(num_types))

            if self.verbose:
                print(f"   Added diversity bonus: λ={self.w_diversity}, {num_types} tool types")

        # Universe size bonus: encourage larger positive universe (tie-breaker)
        universe_size_term = 0
        if self.w_universe > 0:
            universe_size_term = self.w_universe * (
                cp.quicksum(y[i] for i in range(self.G)) +
                cp.quicksum(z[k] for k in range(self.N))
            )
            if self.verbose:
                print(f"   Added universe size bonus: ε={self.w_universe}")

        # Objective: max w_recall·(Recall) - w_fpr·(FPR) + λ·(diversity) + ε·|U+|
        recall_term = (self.w_recall / self.G) * cp.quicksum(y[i] for i in range(self.G)) if self.G > 0 else 0
        fpr_term = (self.w_fpr / self.N) * cp.quicksum(z[k] for k in range(self.N)) if self.N > 0 else 0
        obj = recall_term - fpr_term + diversity_term + universe_size_term
        model.setObjective(obj, COPT.MAXIMIZE)

        # Solve
        model.solve()
        solve_time = time.time() - start_time

        status_map = {COPT.OPTIMAL: "optimal", COPT.TIMEOUT: "timeout", COPT.INFEASIBLE: "infeasible"}
        status_str = status_map.get(model.status, f"status_{model.status}")

        if self.verbose:
            print(f"✅ {status_str} in {solve_time:.2f}s")

        # Check if solution is available
        if model.status != COPT.OPTIMAL:
            if self.verbose:
                print(f"⚠️  No solution available")
            return {
                "selected_tools": [],
                "positive_universe": [],
                "recall": 0.0,
                "precision": 0.0,
                "num_tools": 0,
                "status": status_str,
                "solve_time": solve_time,
                "objective_value": 0.0
            }

        # Extract solution
        selected_idx = [j for j in range(self.T) if x[j].x > 0.5]
        selected_tools = [self.tool_calls[j] for j in selected_idx]

        # Verify min_tools constraint
        if self.min_tools is not None and len(selected_idx) < self.min_tools:
            if self.verbose:
                print(f"⚠️  WARNING: Solution violates min_tools constraint!")
                print(f"   Expected: >= {self.min_tools}, Got: {len(selected_idx)}")

        # Compute U+ (union of all selected tools' results)
        u_plus = set()
        for j in selected_idx:
            u_plus.update(self.tool_results[self.tool_calls[j]["id"]])
        u_plus_sorted = sorted(u_plus)

        # Metrics
        # Recall = |U+ ∩ G| / |G|
        num_covered = len(u_plus & set(self.GT))
        recall = num_covered / self.G if self.G > 0 else 0.0

        # Precision = |U+ ∩ G| / |U+|
        precision = num_covered / len(u_plus) if u_plus else 0.0

        if self.verbose:
            print(f"   Tools={len(selected_tools)}, Recall={recall:.1%}, Precision={precision:.1%}, |U+|={len(u_plus)}")

        return {
            "selected_tools": selected_tools,
            "positive_universe": u_plus_sorted,
            "recall": recall,
            "precision": precision,
            "num_tools": len(selected_tools),
            "status": status_str,
            "solve_time": solve_time,
            "objective_value": model.objval if model.status == COPT.OPTIMAL else 0.0
        }

