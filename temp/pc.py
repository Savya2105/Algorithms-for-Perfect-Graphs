# perfect_coloring.py

from typing import Dict, List, Optional, Set, Tuple
import networkx as nx


def max_clique_branch_and_bound(G: nx.Graph) -> Set[int]:
    """
    Exact maximum clique using a Bron–Kerbosch style branch-and-bound.

    Returns a set of node labels corresponding to one maximum clique.
    """
    best_clique: Set[int] = set()

    def bronk(R: Set[int], P: Set[int], X: Set[int]) -> None:
        nonlocal best_clique
        if not P and not X:
            # Found a maximal clique
            if len(R) > len(best_clique):
                best_clique = set(R)
            return

        # Simple bound: if R ∪ P cannot beat best_clique, prune
        if len(R) + len(P) <= len(best_clique):
            return

        # Pivoting: pick arbitrary pivot from P ∪ X
        if P or X:
            u = next(iter(P or X))
            # Explore only vertices in P \ N(u)
            for v in list(P - set(G.neighbors(u))):
                bronk(
                    R | {v},
                    P & set(G.neighbors(v)),
                    X & set(G.neighbors(v)),
                )
                P.remove(v)
                X.add(v)

    nodes = set(G.nodes())
    bronk(set(), nodes, set())
    return best_clique


def greedy_order(G: nx.Graph) -> List[int]:
    """
    Return a vertex ordering (simple heuristic) for colouring:
    sort by descending degree.
    """
    return sorted(G.nodes(), key=lambda v: G.degree[v], reverse=True)


def colour_with_k_colours(
    G: nx.Graph, k: int
) -> Optional[Dict[int, int]]:
    """
    Try to colour G with exactly k colours using backtracking/branch-and-bound.

    Returns a dict {node: colour} with colours in {0, ..., k-1}
    or None if impossible.
    """
    order = greedy_order(G)
    colour_of: Dict[int, int] = {}
    n = len(order)

    # For each prefix, keep track of used colours to prune
    def backtrack(idx: int) -> bool:
        if idx == n:
            return True

        v = order[idx]

        # Determine forbidden colours from already coloured neighbours
        forbidden = set()
        for u in G.neighbors(v):
            if u in colour_of:
                forbidden.add(colour_of[u])

        for c in range(k):
            if c in forbidden:
                continue
            colour_of[v] = c

            # Optional pruning: if we already used > k colours, fail
            if backtrack(idx + 1):
                return True

            del colour_of[v]

        return False

    success = backtrack(0)
    if success:
        return colour_of
    return None


def perfect_graph_optimal_colouring(G: nx.Graph) -> Tuple[int, Dict[int, int]]:
    """
    Compute an optimal colouring of a perfect graph.

    Strategy:
        1. Compute a maximum clique to get ω(G).
        2. Try to colour G with ω(G) colours using backtracking.
        3. If successful, return χ(G) = ω(G) and the colouring.

    Note: This assumes G is perfect. For non-perfect graphs,
    χ(G) can be > ω(G), and the algorithm may fail.
    """
    max_clique = max_clique_branch_and_bound(G)
    omega = len(max_clique)

    colouring = colour_with_k_colours(G, omega)
    if colouring is None:
        raise RuntimeError(
            "Failed to colour with ω(G) colours. "
            "Either G is not perfect or the search was pruned incorrectly."
        )

    return omega, colouring

if __name__ == "__main__":
    # Example 1: complete graph K_n
    n = 6
    K = nx.complete_graph(n)
    chi, col = perfect_graph_optimal_colouring(K)
    print("χ(K_n) =", chi, "colours used:", len(set(col.values())))

    # Example 2: bipartite graph (perfect), should use 2 colours
    B = nx.complete_bipartite_graph(3, 4)
    chi2, col2 = perfect_graph_optimal_colouring(B)
    print("χ(K_{3,4}) =", chi2, "colours used:", len(set(col2.values())))
