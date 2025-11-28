import cvxpy as cp
import numpy as np

def lovasz_theta(adj_matrix):
    """
    Computes the Lovasz Theta function (theta(G)) using Semidefinite Programming.
    
    Paper: Lovasz, L. (1979). "On the Shannon Capacity of a Graph".
    Method: Dual formulation maximizing Trace(B * J).
    
    Args:
        adj_matrix (np.ndarray): The adjacency matrix of the graph (0s and 1s).
                                 Must be symmetric with 0 on diagonal.
                                 
    Returns:
        float: The value of theta(G).
    """
    n = adj_matrix.shape
    if n == 0:
        return 0.0
    
    # 1. Define the SDP variable: A symmetric n x n matrix
    # PSD=True enforces the positive semidefinite constraint (X >= 0)
    B = cp.Variable((n, n), PSD=True)
    
    # 2. Create the all-ones matrix J
    J = np.ones((n, n))
    
    # 3. Objective: Maximize Trace(B * J), which is equivalent to sum of entries of B
    objective = cp.Maximize(cp.trace(B @ J))
    
    # 4. Constraints
    constraints =
    
    # Constraint A: Trace(B) == 1
    constraints.append(cp.trace(B) == 1)
    
    # Constraint B: B_ij == 0 for all edges (i, j) in G (Orthogonality)
    # We locate edges where adj_matrix is 1.
    # Note: We iterate only the upper triangle to avoid redundant constraints
    rows, cols = np.triu_indices(n, k=1)
    for i, j in zip(rows, cols):
        if adj_matrix[i][j] == 1:
            constraints.append(B[i, j] == 0)
            
    # 5. Define and solve the problem
    prob = cp.Problem(objective, constraints)
    
    try:
        # SCS is a standard splitting conic solver included with cvxpy.
        # We relax tolerance slightly for speed in benchmarking.
        result = prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)
    except cp.error.SolverError:
        try:
            # Fallback if SCS fails
            result = prob.solve(verbose=False)
        except:
            return -1.0

    return result