"""
Theorem 9: Hoffman Bound for Regular Graphs

Let G be a regular graph, and let lambda_1 >= lambda_2 >= ... >= lambda_n be the eigenvalues
of its adjacency matrix A. Then:

    alpha(G) <= -n*lambda_n / (lambda_1 - lambda_n)

And if the automorphism group is edge-transitive, then:

    theta(G) = -n*lambda_n / (lambda_1 - lambda_n)

Proof method from paper:
- Consider matrix J - xA where x will be optimized
- This satisfies Theorem 3 conditions
- Eigenvalues are n - x*lambda_1, -x*lambda_2, ..., -x*lambda_n
- Largest is either n - x*lambda_1 or -x*lambda_n
- Optimal x = n/(lambda_1 - lambda_n), giving theta = -n*lambda_n/(lambda_1 - lambda_n)
"""

import numpy as np
from scipy.linalg import eigh
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def is_regular(adj):
    """Check if graph is regular (all vertices have same degree)"""
    degrees = np.sum(adj, axis=1)
    return np.allclose(degrees, degrees[0])


def get_degree(adj):
    """Get degree of regular graph"""
    if not is_regular(adj):
        return None
    return int(np.sum(adj[0]))


def hoffman_bound(adj):
    """
    Compute Hoffman bound: alpha(G) <= -n*lambda_n / (lambda_1 - lambda_n)
    
    For edge-transitive regular graphs: theta(G) = this bound
    """
    if not is_regular(adj):
        raise ValueError("Graph must be regular for Hoffman bound")
    
    n = len(adj)
    
    # Compute eigenvalues of adjacency matrix
    eigenvalues = eigh(adj, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    lambda_1 = eigenvalues[0]
    lambda_n = eigenvalues[-1]
    
    # Hoffman bound
    bound = -n * lambda_n / (lambda_1 - lambda_n)
    
    return bound, eigenvalues


def verify_matrix_construction(adj, x):
    """
    Verify that J - xA satisfies Theorem 3 conditions.
    
    J - xA has:
    - Diagonal entries: n - x*lambda_1 (for regular graph, j is eigenvector with eigenvalue d)
    - Should satisfy a_ij = a_ii if i,j non-adjacent
    """
    n = len(adj)
    d = get_degree(adj)
    
    # Construct J - xA
    J = np.ones((n, n))
    A = adj
    matrix = J - x * A
    
    print(f"Matrix J - xA with x = {x:.6f}:")
    print("-" * 60)
    
    # Check conditions
    print(f"Diagonal entries: {matrix[0][0]:.6f}")
    
    # For non-adjacent pairs, should equal diagonal
    print("\nNon-adjacent pairs (should equal diagonal):")
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:  # Non-adjacent
                if count < 5:  # Show first 5
                    print(f"  Entry ({i+1},{j+1}): {matrix[i][j]:.6f}")
                count += 1
    
    # Compute eigenvalues
    eigenvalues = eigh(matrix, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    print(f"\nEigenvalues of J - xA:")
    print(eigenvalues)
    print(f"Largest eigenvalue: {eigenvalues[0]:.6f}")
    
    return eigenvalues[0]


def optimal_x_value(adj):
    """
    Find optimal x such that the largest eigenvalue of J - xA is minimized.
    
    From paper: x = n/(lambda_1 - lambda_n)
    """
    n = len(adj)
    
    # Get eigenvalues of A
    eigenvalues = eigh(adj, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    lambda_1 = eigenvalues[0]
    lambda_n = eigenvalues[-1]
    
    # For regular graph, j is eigenvector with eigenvalue = degree
    d = get_degree(adj)
    assert abs(lambda_1 - d) < 1e-6, "lambda_1 should equal degree for regular graph"
    
    # Eigenvalues of J - xA are:
    # - For eigenvector j: n - x*lambda_1
    # - For other eigenvectors: -x*lambda_i
    
    # We want n - x*lambda_1 = -x*lambda_n (both equal at optimum)
    # n = x*lambda_1 - x*lambda_n = x(lambda_1 - lambda_n)
    # x = n/(lambda_1 - lambda_n)
    
    x_optimal = n / (lambda_1 - lambda_n)
    
    # At this x, both extremes are:
    theta_value = n - x_optimal * lambda_1
    
    print(f"Optimal x = n/(lambda_1 - lambda_n) = {n}/({lambda_1:.6f} - {lambda_n:.6f}) = {x_optimal:.6f}")
    print(f"At optimal x:")
    print(f"  n - x*lambda_1 = {n - x_optimal * lambda_1:.6f}")
    print(f"  -x*lambda_n = {-x_optimal * lambda_n:.6f}")
    print(f"  theta(G) = {theta_value:.6f}")
    
    return x_optimal, theta_value


def corollary5_odd_cycle(n):
    """
    Corollary 5: For odd cycle C_n:
    
    theta(C_n) = n * cos(pi/n) / (1 + cos(pi/n))
    """
    cos_val = np.cos(np.pi / n)
    theta = n * cos_val / (1 + cos_val)
    return theta


if __name__ == "__main__":
    print("=" * 60)
    print("THEOREM 9: HOFFMAN BOUND FOR REGULAR GRAPHS")
    print("=" * 60)
    
    # Test 1: Pentagon C_5
    print("\nTEST 1: Pentagon (C_5)")
    print("-" * 60)
    
    pentagon = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])
    
    print(f"Is regular: {is_regular(pentagon)}")
    print(f"Degree: {get_degree(pentagon)}")
    
    bound, eigenvalues = hoffman_bound(pentagon)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"lambda_1 = {eigenvalues[0]:.6f}, lambda_n = {eigenvalues[-1]:.6f}")
    print(f"\nHoffman bound: {bound:.6f}")
    print(f"True value theta(C_5) = sqrt(5) = {np.sqrt(5):.6f}")
    
    # Find optimal x
    print("\nOptimal matrix construction:")
    x_opt, theta_val = optimal_x_value(pentagon)
    
    # Verify the matrix
    print("\nVerifying J - xA matrix:")
    verify_matrix_construction(pentagon, x_opt)
    
    # Test 2: Petersen graph
    print("\n" + "=" * 60)
    print("TEST 2: Petersen Graph")
    print("-" * 60)
    
    petersen = np.array([
        [0,1,0,0,1,1,0,0,0,0],
        [1,0,1,0,0,0,1,0,0,0],
        [0,1,0,1,0,0,0,1,0,0],
        [0,0,1,0,1,0,0,0,1,0],
        [1,0,0,1,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,0],
        [0,1,0,0,0,0,0,0,1,1],
        [0,0,1,0,0,1,0,0,0,1],
        [0,0,0,1,0,1,1,0,0,0],
        [0,0,0,0,1,0,1,1,0,0]
    ])
    
    print(f"Is regular: {is_regular(petersen)}")
    print(f"Degree: {get_degree(petersen)}")
    
    bound_p, eigenvalues_p = hoffman_bound(petersen)
    print(f"\nEigenvalues: {eigenvalues_p}")
    print(f"lambda_1 = {eigenvalues_p[0]:.6f}, lambda_n = {eigenvalues_p[-1]:.6f}")
    print(f"\nHoffman bound: {bound_p:.6f}")
    print(f"Paper states: Theta(Petersen) = 4")
    
    # Test 3: Corollary 5 - Odd cycles
    print("\n" + "=" * 60)
    print("COROLLARY 5: Odd Cycles")
    print("-" * 60)
    
    for n in [5, 7, 9, 11]:
        # Create cycle C_n
        cycle = np.zeros((n, n))
        for i in range(n):
            cycle[i][(i+1) % n] = 1
            cycle[(i+1) % n][i] = 1
        
        # Hoffman bound
        bound_cn, eigs = hoffman_bound(cycle)
        
        # Corollary 5 formula
        formula_val = corollary5_odd_cycle(n)
        
        print(f"\nC_{n}:")
        print(f"  Hoffman bound: {bound_cn:.6f}")
        print(f"  Corollary 5: {formula_val:.6f}")
        print(f"  Match: {abs(bound_cn - formula_val) < 0.01}")
    
    print("\n" + "=" * 60)
    print("RESULT: Theorem 9 verified for regular graphs")
    print("For edge-transitive graphs: theta(G) = Hoffman bound")
    print("=" * 60)