# benchmark.py

import time
from typing import Callable, List, Tuple

import networkx as nx

from lovaz import lovasz_theta_sdp
from pc import perfect_graph_optimal_colouring


def time_call(fn: Callable, *args, **kwargs) -> Tuple[float, object]:
    """Run fn(*args, **kwargs) and return (elapsed_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0, result


def generate_test_graphs(n_values: List[int]) -> List[Tuple[str, nx.Graph]]:
    """
    Generate a small suite of graphs for each n:
    - complete graph
    - cycle graph (C_n)
    - random bipartite (perfect)
    """
    graphs = []
    for n in n_values:
        if n < 3:
            continue

        K = nx.complete_graph(n)
        graphs.append((f"K_{n}", K))

        C = nx.cycle_graph(n)
        graphs.append((f"C_{n}", C))

        # Random bipartite graph with n nodes split ~n/2, n/2
        left = n // 2
        right = n - left
        B = nx.complete_bipartite_graph(left, right)
        graphs.append((f"K_{{{left},{right}}}", B))

    return graphs


def main():
    n_values = [5, 8, 10, 12]  # you can increase once it's stable
    graphs = generate_test_graphs(n_values)

    print("name,n,theta_time,theta_value,color_time,chi")
    for name, G in graphs:
        n = G.number_of_nodes()

        # Time Lovasz theta
        try:
            theta_time, theta_val = time_call(lovasz_theta_sdp, G)
        except Exception as e:
            theta_time, theta_val = float("nan"), None
            print(f"# θ(G) failed for {name}: {e}")

        # Time colouring (perfect graph algorithm)
        try:
            color_time, (chi_val, colouring) = time_call(
                perfect_graph_optimal_colouring, G
            )
        except Exception as e:
            color_time, chi_val = float("nan"), None
            print(f"# colouring failed for {name}: {e}")

        print(
            f"{name},{n},{theta_time:.4f},{theta_val},{color_time:.4f},{chi_val}"
        )


if __name__ == "__main__":
    main()
