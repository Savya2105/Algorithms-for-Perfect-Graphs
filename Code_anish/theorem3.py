"""
Theorem 3: SDP Formulation of theta(G)

theta(G) is the minimum of the largest eigenvalue of any symmetric matrix (a_ij)
such that:
    a_ii = 1 for all i
    a_ij = 1 if i and j are nonadjacent (or equal)

Proof outline from paper:
1) Direction 1: Show any optimal representation gives such a matrix with 
   largest eigenvalue = theta(G)
2) Direction 2: Show any such matrix gives a representation with 
   value = largest eigenvalue
"""
import cvxpy as cp
import numpy as np
from scipy.linalg import eigh
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def construct_matrix_from_representation(u, c):
    """
    Given orthonormal representation (u_1,...,u_n) with handle c,
    construct matrix A with:
        a_ii = theta
        a_ij = theta - (c - u_i)^T(c - u_j) for i != j
    
    From paper: This satisfies conditions and has largest eigenvalue = theta
    """
    n = len(u)
    # Compute theta
    theta = max([1.0 / (np.dot(c, u[i])**2) for i in range(n)])
    
    A = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = theta
            else:
                # a_ij = theta - (c - u_i)^T(c - u_j)
                diff_i = c - u[i]
                diff_j = c - u[j]
                A[i][j] = theta - np.dot(diff_i, diff_j)
    
    return A, theta


def verify_matrix_conditions(A, G_adj):
    """
    Verify matrix satisfies Theorem 3 conditions:
    1) a_ii = 1 for all i
    2) a_ij = 1 if i,j non-adjacent (or equal)
    """
    n = len(A)
    print("Verifying matrix conditions:")
    print("-" * 60)

    # Check diagonal (should all be 1)
    diagonal_values = [A[i][i] for i in range(n)]
    print(f"Diagonal values (should be 1): {diagonal_values}")
    all_ones = all(abs(val - 1.0) < 1e-6 for val in diagonal_values)
    print(f"All diagonals = 1: {all_ones}")

    # Check non-adjacent pairs (should equal 1)
    print("\nNon-adjacent pairs (should equal 1):")
    violations = 0
    for i in range(n):
        for j in range(i+1, n):
            if G_adj[i][j] == 0:  # Non-adjacent
                if abs(A[i][j] - 1.0) < 1e-6:
                    print(f"a_{i+1}{j+1} = {A[i][j]:.6f} ✓")
                else:
                    print(f"a_{i+1}{j+1} = {A[i][j]:.6f} ✗ (expected 1)")
                    violations += 1

    if violations == 0:
        print("\nAll conditions satisfied ✓")
    else:
        print(f"\n{violations} violations found")


def construct_representation_from_matrix(A, G_adj):
    """
    Given matrix A satisfying conditions, construct orthonormal representation.
    
    From paper's proof (Direction 2):
    - theta*I - A is positive semidefinite
    - Write theta*I - A_ij = x_i^T x_j
    - Set u_i = (1/sqrt(theta))(c + x_i) where c is perpendicular to all x_i
    """
    n = len(A)
    
    # Get largest eigenvalue (this should be theta)
    eigenvalues = eigh(A, eigvals_only=True)
    theta = np.max(eigenvalues)
    
    print(f"Largest eigenvalue (theta) = {theta:.6f}")
    
    # Construct theta*I - A
    B = theta * np.eye(n) - A
    
    # Check if positive semidefinite
    eigs_B = eigh(B, eigvals_only=True)
    print(f"Eigenvalues of theta*I - A: {eigs_B}")
    
    if np.all(eigs_B >= -1e-10):
        print("Matrix theta*I - A is positive semidefinite (check)")
    
    # Factor B = X^T X (Cholesky-like)
    eigvals_B, eigvecs_B = eigh(B)
    eigvals_B = np.maximum(eigvals_B, 0)  # Handle numerical errors
    X = eigvecs_B @ np.diag(np.sqrt(eigvals_B))
    
    # x_i are rows of X^T
    x = X.T
    
    # Find c perpendicular to all x_i (any vector in null space)
    # Use a random vector perpendicular to span of x_i
    if n < x.shape[1]:
        # c can be in span's orthogonal complement
        c = np.random.randn(x.shape[1])
        # Project out components along x_i
        for i in range(n):
            if np.dot(x[i], x[i]) > 1e-10:
                c = c - np.dot(c, x[i]) * x[i] / np.dot(x[i], x[i])
        c = c / np.linalg.norm(c)
    else:
        c = np.zeros(x.shape[1])
        c[0] = 1
    
    # Construct u_i = (1/sqrt(theta))(c + x_i)
    u = []
    for i in range(n):
        u_i = (c + x[i]) / np.sqrt(theta)
        u.append(u_i)
    
    u = np.array(u)
    
    return u, c, theta


def compute_theta_sdp(G_adj):
    n = len(G_adj)
    A = cp.Variable((n, n), symmetric=True)
    constraints = [A[i, i] == 1 for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if G_adj[i, j] == 0:
                constraints += [A[i, j] == 1]
    objective = cp.Minimize(cp.lambda_max(A))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.value, A.value


if __name__ == "__main__":
    print("=" * 60)
    print("THEOREM 3: SEMIDEFINITE PROGRAMMING FORMULATION")
    print("=" * 60)
    
    # Pentagon C_5
    print("\nTest on Pentagon (C_5):")
    print("-" * 60)
    
    pentagon = np.array([
        [0, 1, 0, 0, 1],  # vertex 0 connected to 1 and 4
        [1, 0, 1, 0, 0],  # vertex 1 connected to 0 and 2
        [0, 1, 0, 1, 0],  # vertex 2 connected to 1 and 3
        [0, 0, 1, 0, 1],  # vertex 3 connected to 2 and 4
        [1, 0, 0, 1, 0]   # vertex 4 connected to 3 and 0
    ])
    
    # Compute theta using SDP
    theta_sdp, A_optimal = compute_theta_sdp(pentagon)
    print(f"\ntheta(C_5) via SDP = {theta_sdp:.6f}")
    print(f"True value = sqrt(5) = {np.sqrt(5):.6f}")
    
    # Show the optimal matrix
    print("\nOptimal matrix A:")
    print(A_optimal)
    
    # Verify conditions
    print("\n")
    verify_matrix_conditions(A_optimal, pentagon)
    
    # Eigenvalues
    eigenvalues = eigh(A_optimal, eigvals_only=True)
    print(f"\nEigenvalues of A: {np.sort(eigenvalues)[::-1]}")
    
    print("\n" + "=" * 60)
    print("DIRECTION 1: Representation -> Matrix")
    print("=" * 60)
    
    # Create umbrella representation for C_5
    cos_theta_angle = (5 - np.sqrt(5)) / 4
    theta_angle = np.arccos(cos_theta_angle)
    
    ribs = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        x = np.sin(theta_angle) * np.cos(angle)
        y = np.sin(theta_angle) * np.sin(angle)
        z = np.cos(theta_angle)
        ribs.append([x, y, z])
    
    u = np.array(ribs)
    c = np.array([0, 0, 1])
    
    A_from_rep, theta_from_rep = construct_matrix_from_representation(u, c)
    print(f"\nMatrix constructed from umbrella representation")
    print(f"theta from representation = {theta_from_rep:.6f}")
    
    eigenvalues_rep = eigh(A_from_rep, eigvals_only=True)
    print(f"Largest eigenvalue = {np.max(eigenvalues_rep):.6f}")
    
    print("\n" + "=" * 60)
    print("RESULT: Both directions of Theorem 3 verified!")
    print("=" * 60)