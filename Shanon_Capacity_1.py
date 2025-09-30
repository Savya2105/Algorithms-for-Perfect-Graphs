import cvxpy as cp
import numpy as np

def lovasz_theta(adj):
    n = adj.shape[0]
    B = cp.Variable((n, n), symmetric=True)
    J = np.ones((n, n))
    objective = cp.Maximize(cp.trace(B @ J))
    constraints = [B >> 0, cp.trace(B) == 1]
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1 and i != j:
                constraints.append(B[i, j] == 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    return prob.value

# Build adjacency for C5 (pentagon)
n = 7
adj_C5 = np.zeros((n, n))
for i in range(n):
    adj_C5[i, (i+1) % n] = 1
    adj_C5[(i+1) % n, i] = 1

# Call the function
theta_val = lovasz_theta(adj_C5)
print(f"Lovasz theta of C5:", theta_val)
