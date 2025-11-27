import itertools
import math
import numpy as np
import time


try:
    import cvxpy as cp
except Exception:
    cp = None


# ============================
# Graph structure and utilities
# ============================

class Graph:
    def __init__(self, n, edges=None):
        self.n = n
        self.adj = np.zeros((n, n), dtype=int)
        if edges:
            for (i, j) in edges:
                self.add_edge(i, j)

    def add_edge(self, i, j):
        if i == j:
            return
        self.adj[i, j] = 1
        self.adj[j, i] = 1

    def is_edge(self, i, j):
        return bool(self.adj[i, j])

    def edges(self):
        n = self.n
        return [(i, j) for i in range(n) for j in range(i + 1, n) if self.adj[i, j]]

    def adjacency_matrix(self):
        return self.adj.copy()

    def complement(self):
        n = self.n
        Gc = Graph(n)
        for i in range(n):
            for j in range(i + 1, n):
                if not self.adj[i, j]:
                    Gc.add_edge(i, j)
        return Gc

    def strong_product(self, other):
        """
        Strong (AND) product G ⊠ H:
        (a,b) adjacent to (c,d) iff
        (a==c or a~c) and (b==d or b~d), and not identical.
        """
        n1, n2 = self.n, other.n
        prod = Graph(n1 * n2)
        for a in range(n1):
            for b in range(n2):
                for c in range(n1):
                    for d in range(n2):
                        if (a == c or self.adj[a, c]) and (b == d or other.adj[b, d]):
                            if not (a == c and b == d):
                                prod.add_edge(a * n2 + b, c * n2 + d)
        return prod

    def __repr__(self):
        return f"Graph(n={self.n}, edges={len(self.edges())})"


# ========================================
# Independence number (brute-force version)
# ========================================

def brute_force_independence_number(G):
    """
    Compute the independence number alpha(G) by exhaustive search.
    Suitable for small graphs (n <= ~24).
    Prints the time taken.
    """
    start = time.perf_counter()
    n = G.n
    for r in range(n, 0, -1):
        for comb in itertools.combinations(range(n), r):
            if all(not G.is_edge(i, j) for i, j in itertools.combinations(comb, 2)):
                elapsed = time.perf_counter() - start
                print(f"alpha(G) = {r} found in {elapsed:.4f}s")
                return r
    elapsed = time.perf_counter() - start
    print(f"No independent set found (elapsed {elapsed:.4f}s)")
    return 0


# ===================================
# Lovasz theta function via SDP (CVXPY)
# ===================================

def lovasz_theta(G, solver='SCS', verbose=False):
    """
    Compute the Lovasz theta value theta(G) via semidefinite programming.
    maximize Tr(JB)
    subject to B PSD, Tr(B)=1, and B_ij=0 for adjacent i,j
    Prints computation time.
    """
    if cp is None:
        raise RuntimeError("cvxpy not installed. Run `pip install cvxpy`.")

    start = time.perf_counter()
    n = G.n
    B = cp.Variable((n, n), PSD=True)
    J = np.ones((n, n))
    constraints = [cp.trace(B) == 1]

    for i in range(n):
        for j in range(i + 1, n):
            if G.is_edge(i, j):
                constraints.append(B[i, j] == 0)
                constraints.append(B[j, i] == 0)

    objective = cp.Maximize(cp.trace(J @ B))
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=solver, verbose=verbose)

    elapsed = time.perf_counter() - start
    theta_val = float(result) if result is not None else None
    print(f"theta(G) ~ {theta_val:.6f} computed in {elapsed:.4f}s (n={n}, solver={solver})")
    return theta_val


# ======================
# Example graph builders
# ======================

def cycle_graph(n):
    """Construct an n-cycle C_n"""
    G = Graph(n)
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    return G


def petersen_graph():
    """Construct the 10-vertex Petersen graph"""
    G = Graph(10)
    # outer 5-cycle
    for i in range(5):
        G.add_edge(i, (i + 1) % 5)
    # inner 5-cycle with skip-2 pattern
    for (a, b) in [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]:
        G.add_edge(a, b)
    # spokes
    for i in range(5):
        G.add_edge(i, 5 + i)
    return G


# =========================
# Example usage / main test
# =========================

if __name__ == "__main__":
    print("Timing Lovasz theta and independence number computations\n")

    # C5
    print("=== Cycle Graph C5 ===")
    C5 = cycle_graph(5)
    alpha_c5 = brute_force_independence_number(C5)
    if cp:
        theta_c5 = lovasz_theta(C5)
        print(f"Expected sqrt(5) ~ {math.sqrt(5):.6f}\n")

    # Petersen
    print("=== Petersen Graph ===")
    P = petersen_graph()
    alpha_p = brute_force_independence_number(P)
    if cp:
        theta_p = lovasz_theta(P)
        print(f"Expected 4.000000\n")
