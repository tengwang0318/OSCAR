"""Stage 2 F-beta MIP Solver: Logical composition using Dinkelbach iteration.

Note: This solver only supports intersection and union operations.
      Difference (negation) operations should be handled by planner in Stage 1.
"""

import time
from typing import List, Dict, Set
import numpy as np
import coptpy as cp
from coptpy import COPT

from .schemas import Clause


class Stage2FbetaSolver:
    """Stage 2 F-beta MIP Solver: Optimize F-beta using Dinkelbach iteration.

    Variables:
    - u[c]: binary, whether clause c is selected
    - y[i]: binary, whether GT image i is covered
    - z[k]: binary, whether non-GT image k is covered
    - type_used[t]: binary, whether tool type t is used (if diversity_weight > 0)

    Constraints:
    - GT coverage: y[i] = 1 iff at least one selected clause covers GT image i
    - Non-GT coverage: z[k] = 1 iff at least one selected clause covers non-GT image k
    - Clause budget: sum(u[c]) <= max_clauses
    - Tool type diversity (if diversity_weight > 0):
        - type_used[t] = 1 iff at least one selected clause uses tool type t
        - For each tool type t: type_used[t] <= sum(u[c] for c in clauses_using_type[t])
        - For each tool type t: sum(u[c] for c in clauses_using_type[t]) <= M * type_used[t]

    Objective:
    Φ_λ = (1+β²)·TP - λ·(β²·|G| + TP + FP) - α·complexity + γ·diversity + δ·negation_used
    where:
    - TP = sum(y[i]) (true positives: covered GT images)
    - FP = sum(z[k]) (false positives: covered non-GT images)
    - complexity = sum(clause_complexity[c] * u[c]) (total tools in selected clauses)
    - diversity = sum(type_used[t]) (number of distinct tool types used)
    - negation_used = 1 if any selected clause contains a negation tool, 0 otherwise
    """

    def __init__(self, clauses: List[Clause], A_matrix: np.ndarray, B_matrix: np.ndarray,
                 num_gt: int, max_clauses, eps, max_iter, beta,
                 complexity_penalty, diversity_weight, negation_weight,
                 time_limit=None, verbose=True):
        self.clauses = clauses
        self.A = A_matrix
        self.B = B_matrix
        self.G = num_gt
        self.N = B_matrix.shape[0]
        self.C = len(clauses)
        self.max_clauses = max_clauses
        self.eps = eps
        self.max_iter = max_iter
        self.beta = beta
        self.complexity_penalty = complexity_penalty
        self.diversity_weight = diversity_weight
        self.negation_weight = negation_weight
        self.time_limit = time_limit
        self.verbose = verbose
        self.d_A = A_matrix.sum(axis=1)
        self.d_B = B_matrix.sum(axis=1)

        # Precompute clause complexity (number of tools in each clause)
        self.clause_complexity = [len(c.positive_tools) + len(c.negative_tools)
                                  for c in clauses]

        # Precompute tool type diversity information
        self.tool_types = {}  # tool_name -> type_index
        self.clause_tool_types = []  # clause_index -> set of tool type indices
        self._build_tool_type_mapping()

        # Precompute negation tool information for each clause
        self.clause_negation_tools = []  # clause_index -> list of negation tool indices in that clause
        self._build_negation_mapping()

    def _build_tool_type_mapping(self):
        """Build mapping from tool types to indices and track which clauses use which types.

        For each clause, we collect all unique tool types used in that clause.
        This allows us to add diversity bonus based on the number of distinct tool types.
        """
        # Collect all unique tool types across all clauses
        all_tool_types = set()
        for clause in self.clauses:
            for tool in clause.positive_tools + clause.negative_tools:
                # tool is a dict, access the "name" key
                all_tool_types.add(tool["name"])

        # Create mapping: tool_name -> type_index
        self.tool_types = {name: idx for idx, name in enumerate(sorted(all_tool_types))}

        # For each clause, collect the set of tool type indices it uses
        self.clause_tool_types = []
        for clause in self.clauses:
            types_in_clause = set()
            for tool in clause.positive_tools + clause.negative_tools:
                # tool is a dict, access the "name" key
                types_in_clause.add(self.tool_types[tool["name"]])
            self.clause_tool_types.append(types_in_clause)

    def _build_negation_mapping(self):
        """Build mapping of negation tools for each clause.

        For each clause, identify which tools are negation tools (is_negation=True).
        This allows us to:
        1. Add negation usage bonus to objective
        2. Enforce constraint: at most 1 negation tool per clause
        """
        self.clause_negation_tools = []
        for clause in self.clauses:
            negation_tool_indices = []
            for idx, tool in enumerate(clause.positive_tools):
                if tool.get("is_negation", False):
                    negation_tool_indices.append(idx)
            self.clause_negation_tools.append(negation_tool_indices)

    def solve(self) -> Dict:
        """Solve using Dinkelbach iteration."""
        if self.verbose:
            diversity_info = f", diversity_weight={self.diversity_weight}" if self.diversity_weight > 0 else ""
            negation_info = f", negation_weight={self.negation_weight}" if self.negation_weight > 0 else ""
            print(f"📊 Solver: C={self.C}, G={self.G}, N={self.N}, M_max={self.max_clauses}{diversity_info}{negation_info}")

        start = time.time()
        lam = 0.0
        best = None

        for it in range(self.max_iter):
            res = self._solve_one_iteration(lam)

            if res["status"] != "optimal":
                if self.verbose:
                    print(f"⚠️  Iter {it}: {res['status']}")
                break

            TP, FP, phi = res["TP"], res["FP"], res["phi_star"]

            if self.verbose:
                print(f"   Iter {it}: λ={lam:.6f}, TP={TP}, FP={FP}, Φ*={phi:.6f}")

            # Convergence check
            if abs(phi) <= self.eps:
                best = res
                if self.verbose:
                    print(f"✅ Converged: |Φ*| ≤ {self.eps}")
                break

            # Update lambda: F_beta = (1+beta^2)*TP / (beta^2*G + TP + FP)
            beta2 = self.beta ** 2
            denom = beta2 * self.G + TP + FP
            lam_new = ((1 + beta2) * TP) / denom if denom > 0 else 0.0

            if abs(lam_new - lam) <= self.eps:
                best = res
                if self.verbose:
                    print(f"✅ Converged: |Δλ| ≤ {self.eps}")
                break

            lam = lam_new
            best = res

        t = time.time() - start

        if not best:
            return {"selected_clause_indices": [], "TP": 0, "FP": 0,
                   "recall": 0.0, "precision": 0.0, "f1": 0.0,
                   "iterations": it + 1, "lambda_final": lam, "phi_final": 0.0,
                   "status": "failed", "solve_time": t}

        TP, FP = best["TP"], best["FP"]
        beta2 = self.beta ** 2
        denom = beta2 * self.G + TP + FP
        f_beta = ((1 + beta2) * TP) / denom if denom > 0 else 0.0

        # Calculate recall, precision, f1
        recall = TP / self.G if self.G > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if self.verbose:
            print(f"✅ Solved in {t:.2f}s, F{self.beta}={f_beta:.4f}")

        return {
            "selected_clause_indices": best["selected_clause_indices"],
            "TP": TP, "FP": FP,
            "recall": recall, "precision": precision, "f1": f1,
            "iterations": it + 1,
            "lambda_final": lam,
            "phi_final": best["phi_star"],
            "status": "optimal",
            "solve_time": t
        }
    
    def _solve_one_iteration(self, lam: float) -> Dict:
        """Solve one Dinkelbach iteration."""
        model = cp.Envr().createModel("stage2")
        model.setParam(COPT.Param.Logging, 0)
        if self.time_limit:
            model.setParam(COPT.Param.TimeLimit, self.time_limit)

        # Variables
        u = model.addVars(self.C, vtype=COPT.BINARY)
        y = model.addVars(self.G, vtype=COPT.BINARY)
        z = model.addVars(self.N, vtype=COPT.BINARY)

        # GT coverage (OR-linearization)
        for i in range(self.G):
            if self.d_A[i] > 0:
                model.addConstr(y[i] <= cp.quicksum(self.A[i, c] * u[c] for c in range(self.C)))
                model.addConstr(cp.quicksum(self.A[i, c] * u[c] for c in range(self.C)) <= self.d_A[i] * y[i])
            else:
                model.addConstr(y[i] == 0)

        # Non-GT coverage (OR-linearization)
        for k in range(self.N):
            if self.d_B[k] > 0:
                model.addConstr(z[k] <= cp.quicksum(self.B[k, c] * u[c] for c in range(self.C)))
                model.addConstr(cp.quicksum(self.B[k, c] * u[c] for c in range(self.C)) <= self.d_B[k] * z[k])
            else:
                model.addConstr(z[k] == 0)

        # Clause budget
        model.addConstr(cp.quicksum(u[c] for c in range(self.C)) <= self.max_clauses)

        # Subset exclusion: if clause_j's result_set ⊂ clause_i's result_set, don't select both
        # Because: SuperSet ∪ SubSet = SuperSet (subset is redundant in UNION)
        subset_exclusions = 0
        for i in range(self.C):
            set_i = set(self.clauses[i].result_set)
            for j in range(i + 1, self.C):
                set_j = set(self.clauses[j].result_set)
                # Check if one is a proper subset of the other
                if set_i < set_j:  # i is proper subset of j
                    model.addConstr(u[i] + u[j] <= 1, name=f"subset_excl_{i}_{j}")
                    subset_exclusions += 1
                elif set_j < set_i:  # j is proper subset of i
                    model.addConstr(u[i] + u[j] <= 1, name=f"subset_excl_{j}_{i}")
                    subset_exclusions += 1
        
        if self.verbose and subset_exclusions > 0:
            print(f"   Added {subset_exclusions} subset exclusion constraints")

        # Negation usage bonus: add bonus if any negation tool is used
        # Note: The constraint "at most 1 negation per clause" is enforced during clause generation
        negation_term = 0
        if self.negation_weight > 0:
            # Check if any clause uses negation tools
            clauses_with_negation = [c for c in range(self.C) if len(self.clause_negation_tools[c]) > 0]

            if clauses_with_negation:
                # Binary variable: 1 if any negation tool is used, 0 otherwise
                negation_used = model.addVar(vtype=COPT.BINARY, name="negation_used")

                # negation_used = 1 iff at least one clause with negation is selected
                # Constraint 1: negation_used <= sum(u[c] for c in clauses_with_negation)
                model.addConstr(negation_used <= cp.quicksum(u[c] for c in clauses_with_negation))

                # Constraint 2: sum(u[c] for c in clauses_with_negation) <= M * negation_used
                M = len(clauses_with_negation)
                model.addConstr(cp.quicksum(u[c] for c in clauses_with_negation) <= M * negation_used)

                # Add negation bonus to objective
                negation_term = self.negation_weight * negation_used

        # Tool type diversity variables and constraints
        diversity_term = 0
        if self.diversity_weight > 0 and self.tool_types:
            num_types = len(self.tool_types)
            type_used = model.addVars(num_types, vtype=COPT.BINARY, nameprefix="type_used")

            # For each tool type, link type_used[t] to clause selection
            for tool_name, type_idx in self.tool_types.items():
                # Find all clauses that use this tool type
                clauses_using_type = [c for c in range(self.C) if type_idx in self.clause_tool_types[c]]

                if clauses_using_type:
                    # type_used[t] = 1 iff at least one clause using type t is selected
                    # Constraint 1: type_used[t] <= sum(u[c] for c in clauses_using_type)
                    model.addConstr(type_used[type_idx] <= cp.quicksum(u[c] for c in clauses_using_type))

                    # Constraint 2: sum(u[c] for c in clauses_using_type) <= M * type_used[t]
                    # where M = number of clauses using this type
                    M = len(clauses_using_type)
                    model.addConstr(cp.quicksum(u[c] for c in clauses_using_type) <= M * type_used[type_idx])

            # Diversity term: number of distinct tool types used
            diversity_term = self.diversity_weight * cp.quicksum(type_used[t] for t in range(num_types))

        # Objective: Φ_λ = (1+β²)·TP - λ·(β²·|G| + TP + FP) - α·(total tools) + γ·(tool types) + δ·(has negation)
        beta2 = self.beta ** 2
        TP = cp.quicksum(y[i] for i in range(self.G))
        FP = cp.quicksum(z[k] for k in range(self.N))

        # Complexity penalty: sum of tool calls in selected clauses
        complexity = cp.quicksum(self.clause_complexity[c] * u[c] for c in range(self.C))

        obj = (1 + beta2) * TP - lam * (beta2 * self.G + TP + FP) - self.complexity_penalty * complexity + diversity_term + negation_term
        model.setObjective(obj, COPT.MAXIMIZE)

        model.solve()

        if model.status != COPT.OPTIMAL:
            status_map = {COPT.TIMEOUT: "timeout", COPT.INFEASIBLE: "infeasible"}
            return {"status": status_map.get(model.status, f"status_{model.status}")}

        # Extract solution
        TP_val = sum(y[i].x for i in range(self.G))
        FP_val = sum(z[k].x for k in range(self.N))

        return {
            "status": "optimal",
            "selected_clause_indices": [c for c in range(self.C) if u[c].x > 0.5],
            "TP": TP_val,
            "FP": FP_val,
            "phi_star": (1 + beta2) * TP_val - lam * (beta2 * self.G + TP_val + FP_val)
        }

