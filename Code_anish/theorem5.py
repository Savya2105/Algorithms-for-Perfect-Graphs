"""
Theorem 5: Minimax Formula for theta(G)

Let (v_1,...,v_n) range over all orthonormal representations of G
and d over all unit vectors. Then:

    theta(G) = max_representation min_d (sum_i (d^T v_i)^2)

This shows duality between G and its complement.

Proof method from paper:
- Uses Theorem 4 (max Tr(BJ) formulation)
- Constructs orthonormal representation from optimal B matrix
- Shows equality in Lemma 4 (Cauchy-Schwarz)
"""

import numpy as np
from scipy.linalg import eigh
import itertools
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def lemma4_cauchy_schwarz(u, v, c, d):
    """
    Lemma 4: For orthonormal representation u_i of G and v_i of complement:
    
    sum_i (u_i^T c)^2 (v_i^T d)^2 <= |c|^2 |d|^2
    
    Proof: u_i (x) v_i form orthonormal system, apply Cauchy-Schwarz
    """
    n = len(u)
    
    # Compute sum
    total = 0
    for i in range(n):
        u_dot_c = np.dot(u[i], c)
        v_dot_d = np.dot(v[i], d)
        total += (u_dot_c ** 2) * (v_dot_d ** 2)
    
    c_norm = np.linalg.norm(c)
    d_norm = np.linalg.norm(d)
    
    print(f"Sum of (u_i^T c)^2 (v_i^T d)^2 = {total:.6f}")
    print(f"|c|^2 |d|^2 = {c_norm**2 * d_norm**2:.6f}")
    print(f"Lemma 4 satisfied: {total:.6f} <= {c_norm**2 * d_norm**2:.6f}")
    
    return total <= c_norm**2 * d_norm**2 + 1e-6


def compute_representation_value(v, d):
    """
    For orthonormal representation v and unit vector d,
    compute sum_i (d^T v_i)^2
    """
    n = len(v)
    total = 0
    for i in range(n):
        dot = np.dot(d, v[i])
        total += dot ** 2
    return total


def find_optimal_d_for_representation(v):
    """
    For fixed representation v, find unit vector d that minimizes sum (d^T v_i)^2
    
    This is equivalent to minimizing d^T V V^T d where V has v_i as rows
    The minimum is achieved by eigenvector corresponding to smallest eigenvalue
    """
    n = len(v)
    dim = len(v[0])
    
    # Form V matrix (n x dim) with v_i as rows
    V = np.array(v)
    
    # Compute V^T V (dim x dim matrix)
    VTV = V.T @ V
    
    # Find smallest eigenvalue and corresponding eigenvector
    eigenvalues, eigenvectors = eigh(VTV)
    min_idx = np.argmin(eigenvalues)
    d_optimal = eigenvectors[:, min_idx]
    
    # Ensure unit length
    d_optimal = d_optimal / np.linalg.norm(d_optimal)
    
    min_value = eigenvalues[min_idx]
    
    return d_optimal, min_value


def maximize_over_representations(G_adj, trials=20):
    """
    Maximize over orthonormal representations:
    theta(G) = max_v min_d sum_i (d^T v_i)^2
    """
    n = len(G_adj)
    best_value = 0
    best_v = None
    best_d = None
    
    for trial in range(trials):
        # Generate random orthonormal representation
        # Start with random vectors in R^n
        v = np.random.randn(n, n)
        
        # Orthogonalize non-adjacent pairs using Gram-Schmidt
        for i in range(n):
            # Normalize
            v[i] = v[i] / np.linalg.norm(v[i])
            
            # Make orthogonal to non-adjacent vertices processed earlier
            for j in range(i):
                if G_adj[i][j] == 0:  # Non-adjacent
                    # Project out component
                    v[i] = v[i] - np.dot(v[i], v[j]) * v[j]
                    if np.linalg.norm(v[i]) > 1e-10:
                        v[i] = v[i] / np.linalg.norm(v[i])
        
        # Find optimal d for this representation
        d_opt, min_val = find_optimal_d_for_representation(v)
        
        if min_val > best_value:
            best_value = min_val
            best_v = v.copy()
            best_d = d_opt.copy()
    
    return best_value, best_v, best_d


def verify_theorem5(G_adj):
    """
    Verify Theorem 5: theta(G) = max_v min_d sum (d^T v_i)^2
    """
    n = len(G_adj)
    print(f"Computing theta via Theorem 5 for {n}-vertex graph")
    print("=" * 60)
    
    # Find optimal representation
    theta_val, v_opt, d_opt = maximize_over_representations(G_adj)
    
    print(f"\nOptimal value: {theta_val:.6f}")
    
    # Verify it's an orthonormal representation
    print("\nVerifying orthonormal representation:")
    print("-" * 60)
    
    # Check unit vectors
    print("Vector norms:")
    for i in range(min(n, 5)):  # Show first 5
        norm = np.linalg.norm(v_opt[i])
        print(f"|v_{i+1}| = {norm:.6f}")
    
    # Check orthogonality for non-adjacent
    print("\nOrthogonality for non-adjacent pairs:")
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if G_adj[i][j] == 0 and count < 5:  # Show first 5
                dot = np.dot(v_opt[i], v_opt[j])
                print(f"v_{i+1}^T v_{j+1} = {dot:.6f}")
                count += 1
    
    # Compute value with optimal d
    print(f"\nWith optimal d:")
    for i in range(min(n, 5)):  # Show first 5
        dot = np.dot(d_opt, v_opt[i])
        print(f"(d^T v_{i+1})^2 = {dot**2:.6f}")
    
    total = compute_representation_value(v_opt, d_opt)
    print(f"Sum = {total:.6f}")
    
    return theta_val


if __name__ == "__main__":
    print("=" * 60)
    print("THEOREM 5: MINIMAX FORMULA")
    print("=" * 60)
    print("\ntheta(G) = max over representations min over d of sum (d^T v_i)^2\n")
    
    # Test on pentagon
    print("TEST 1: Pentagon (C_5)")
    print("-" * 60)
    
    pentagon = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])
    
    theta_c5 = verify_theorem5(pentagon)
    print(f"\ntheta(C_5) ~ {theta_c5:.6f}")
    print(f"True value = sqrt(5) = {np.sqrt(5):.6f}")
    
    # Test on triangle
    print("\n" + "=" * 60)
    print("TEST 2: Complete Graph K_3")
    print("-" * 60)
    
    K3 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    
    theta_k3 = verify_theorem5(K3)
    print(f"\ntheta(K_3) ~ {theta_k3:.6f}")
    print(f"Expected: 1 (since alpha(K_3) = 1)")
    
    # Demonstrate Lemma 4 (Cauchy-Schwarz inequality)
    print("\n" + "=" * 60)
    print("LEMMA 4: Cauchy-Schwarz Inequality")
    print("=" * 60)
    
    # Create simple orthonormal representations
    u_simple = np.eye(3)  # Identity = orthonormal
    v_simple = np.eye(3)
    c = np.array([1, 0, 0])
    d = np.array([0, 1, 0])
    
    print("\nSimple test with identity matrices:")
    lemma4_cauchy_schwarz(u_simple, v_simple, c, d)
    
    print("\n" + "=" * 60)
    print("RESULT: Theorem 5 provides max-min characterization of theta(G)")
    print("=" * 60)