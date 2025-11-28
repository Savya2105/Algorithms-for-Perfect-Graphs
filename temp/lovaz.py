# lovasz_theta.py

import cvxpy as cp
import numpy as np
import networkx as nx


def lovasz_theta_sdp(G: nx.Graph, solver: str = "SCS", verbose: bool = False) -> float:
    """
    Compute (one standard SDP formulation of) the Lovász theta number θ(G).

    Parameters
    ----------
    G : nx.Graph
        Simple, undirected graph with nodes labelled arbitrarily (we relabel internally).
    solver : str
        CVXPY solver name, e.g. "SCS", "MOSEK", "CVXOPT". SCS is free & easy to install.
    verbose : bool
        If True, forwards solver output.

    Returns
    -------
    float
        Approximate value of θ(G).
    """
    # Relabel nodes to 0..n-1 for convenience
    H = nx.convert_node_labels_to_integers(G, label_attribute="old_label")
    n = H.number_of_nodes()

    # J is the all-ones matrix
    J = np.ones((n, n))

    # Symmetric matrix variable B
    B = cp.Variable((n, n), symmetric=True)

    constraints = []

    # PSD constraint
    constraints.append(B >> 0)

    # Trace constraint
    constraints.append(cp.trace(B) == 1)

    # Edge constraints: for each edge (i,j), B_ij = 0
    for i, j in H.edges():
        constraints.append(B[i, j] == 0)
        constraints.append(B[j, i] == 0)  # redundant but explicit

    # Objective: maximize sum of all entries of B, i.e. <J, B>
    objective = cp.Maximize(cp.sum(cp.multiply(J, B)))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"SDP did not converge: status={prob.status}")

    return float(prob.value)

if __name__ == "__main__":
    import networkx as nx

    # C5 cycle (pentagon) – classic Lovász example
    G = nx.cycle_graph(5)
    theta = lovasz_theta_sdp(G)
    print("θ(C5) ≈", theta)  # should be close to sqrt(5) ≈ 2.236...
