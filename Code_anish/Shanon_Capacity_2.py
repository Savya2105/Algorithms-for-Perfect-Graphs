import numpy as np
from itertools import combinations

def theta_via_eig_scaling(adj, tol=1e-6, max_iter=200):
    n = adj.shape[0]
    J = np.ones((n, n))
    A = adj.copy()
    def lam_max(x):
        M = J - x * A
        vals = np.linalg.eigvalsh(M)
        return vals[-1]
    rho = max(1e-6, np.max(np.abs(np.linalg.eigvalsh(A))))
    x_lo, x_hi = 0.0, 10.0 / rho
    phi = (np.sqrt(5) - 1) / 2
    a, b = x_lo, x_hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = lam_max(c), lam_max(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc > fd:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = lam_max(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = lam_max(c)
        it += 1
    x_opt = (a + b) / 2
    theta_approx = lam_max(x_opt)
    return theta_approx, x_opt, it

def cycle_adj(n):
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i, (i+1) % n] = 1
        adj[(i+1) % n, i] = 1
    return adj

def petersen_adj():
    verts = list(combinations(range(5), 2))
    n = len(verts)
    adj = np.zeros((n, n))
    for i, u in enumerate(verts):
        for j, v in enumerate(verts):
            if i < j and len(set(u).intersection(v)) == 0:
                adj[i, j] = adj[j, i] = 1
    return adj

# Examples
adj_C5 = cycle_adj(5)
adj_petersen = petersen_adj()

theta_C5_approx, xC5, itC5 = theta_via_eig_scaling(adj_C5)
theta_p_approx, xP, itP = theta_via_eig_scaling(adj_petersen)

print("C5 theta approx:", theta_C5_approx, "x_opt:", xC5, "iterations:", itC5)
print("Petersen theta approx:", theta_p_approx, "x_opt:", xP, "iterations:", itP)
