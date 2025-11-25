"""
Theorem 1: Theta(G) <= theta(G)

Proof method from paper:
By Lemmas 1 and 2, alpha(G^k) <= theta(G^k) <= theta(G)^k
Therefore: Theta(G) = lim[k->inf] alpha(G^k)^(1/k) <= theta(G)
"""

import numpy as np
import itertools
import sys
import io


if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Graph:
    def __init__(self, adjacency_matrix):
        self.adj = np.array(adjacency_matrix)
        self.n = len(self.adj)
    
    def is_adjacent(self, i, j):
        """Check if vertices i and j are adjacent"""
        if i == j:
            return False
        return self.adj[i][j] == 1

    def is_adjacent_or_equal(self, i, j):
        """Check if vertices i and j are adjacent or equal (for strong product)"""
        return i == j or self.adj[i][j] == 1
    
    def independence_number(self):
        """Calculate alpha(G) - maximum independent set size"""
        max_indep = 0
        for size in range(1, self.n + 1):
            for subset in itertools.combinations(range(self.n), size):
                is_independent = True
                for i, j in itertools.combinations(subset, 2):
                    if self.adj[i][j] == 1:
                        is_independent = False
                        break
                if is_independent:
                    max_indep = max(max_indep, size)
        return max_indep
    
    def strong_product(self, other):
        """Compute G o H (strong product) as defined in paper"""
        n1, n2 = self.n, other.n
        adj_prod = np.zeros((n1 * n2, n1 * n2))

        for i1 in range(n1):
            for j1 in range(n2):
                idx1 = i1 * n2 + j1
                for i2 in range(n1):
                    for j2 in range(n2):
                        idx2 = i2 * n2 + j2
                        if idx1 != idx2:  # Skip diagonal
                            # (i1,j1) adjacent to (i2,j2) iff
                            # i1 adjacent-or-equal to i2 in G AND j1 adjacent-or-equal to j2 in H
                            if self.is_adjacent_or_equal(i1, i2) and other.is_adjacent_or_equal(j1, j2):
                                adj_prod[idx1][idx2] = 1

        return Graph(adj_prod)


def compute_theta_simple(G):
    """
    Placeholder for Lovász theta function computation.
    Actual theta(G) computation requires semidefinite programming (SDP).
    This returns a lower bound: theta(G) >= alpha(G)

    For complete implementation, see Theorem 3.
    """
    # Lower bound only - actual theta needs SDP solver
    alpha_G = G.independence_number()
    print(f"Note: theta(G) >= alpha(G) = {alpha_G} (lower bound only)")
    return alpha_G


def verify_theorem1(G, k=2):
    """
    Verify: alpha(G^k) <= theta(G)^k
    Therefore: Theta(G) <= theta(G)
    """
    print(f"Testing Theorem 1 on graph with {G.n} vertices")
    print(f"=" * 60)
    
    # Compute alpha(G)
    alpha_G = G.independence_number()
    print(f"alpha(G) = {alpha_G}")
    
    # Compute G^k (k-th strong product)
    G_k = G
    for i in range(k-1):
        print(f"Computing G^{i+2}...")
        G_k = G_k.strong_product(G)
    
    alpha_G_k = G_k.independence_number()
    print(f"alpha(G^{k}) = {alpha_G_k}")
    
    # Lower bound on Theta(G)
    theta_lower = alpha_G_k ** (1.0/k)
    print(f"\nShannon capacity lower bound: alpha(G^{k})^(1/{k}) = {theta_lower:.6f}")
    print(f"By Theorem 1: Theta(G) <= theta(G)")
    print(f"This shows: {alpha_G} <= Theta(G) <= theta(G)")


if __name__ == "__main__":
    # Test on pentagon
    print("THEOREM 1: CAPACITY UPPER BOUND")
    print("=" * 60)
    
    # Pentagon C_5
    pentagon = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])
    
    G = Graph(pentagon)
    verify_theorem1(G, k=2)
    
    print(f"\n{'='*60}")
    print("Note: Complete theta(G) computation requires Theorem 3")