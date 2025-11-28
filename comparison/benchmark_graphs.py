"""
Benchmark Graph Generation Module
==================================
Comprehensive collection of benchmark graphs for testing and comparing
Paper 1 (Lovász Theta) and Paper 2 (Perfect Graph Coloring) algorithms.

Includes:
- Classical graphs (cycles, complete, bipartite)
- Perfect graph classes (interval, chordal, comparability)
- Special graphs mentioned in papers (Pentagon C5, Petersen)
- Structured graphs (random perfect graphs, grid-like)
"""

from graph_core import Graph
import math


class BenchmarkGraphs:
    """Factory class for generating benchmark graphs."""

    @staticmethod
    def create_cycle(n):
        """Create cycle graph Cn."""
        g = Graph()
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        return g

    @staticmethod
    def create_complete(n):
        """Create complete graph Kn."""
        g = Graph()
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j)
        return g

    @staticmethod
    def create_complete_bipartite(n1, n2):
        """Create complete bipartite graph Kn1,n2."""
        g = Graph()
        for i in range(n1):
            for j in range(n1, n1 + n2):
                g.add_edge(i, j)
        return g

    @staticmethod
    def create_pentagon():
        """Create C5 (pentagon) - mentioned in Paper 1."""
        g = Graph()
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in edges:
            g.add_edge(u, v)
        return g

    @staticmethod
    def create_petersen():
        """Create Petersen graph - mentioned in Paper 1 with θ=4."""
        g = Graph()
        # Outer pentagon vertices: 0-4
        # Inner pentagram vertices: 5-9
        # Outer edges
        for i in range(5):
            g.add_edge(i, (i + 1) % 5)
        # Inner edges (pentagram)
        for i in range(5):
            g.add_edge(5 + i, 5 + (i + 2) % 5)
        # Spoke edges
        for i in range(5):
            g.add_edge(i, 5 + i)
        return g

    @staticmethod
    def create_path(n):
        """Create path graph Pn."""
        g = Graph()
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        return g

    @staticmethod
    def create_star(n):
        """Create star graph Sn (one central vertex connected to n-1 others)."""
        g = Graph()
        for i in range(1, n):
            g.add_edge(0, i)
        return g

    @staticmethod
    def create_wheel(n):
        """Create wheel graph: cycle Cn with central hub connected to all."""
        g = Graph()
        # Create cycle
        for i in range(n - 1):
            g.add_edge(i, (i + 1) % (n - 1))
        # Connect hub (vertex n-1) to all cycle vertices
        for i in range(n - 1):
            g.add_edge(n - 1, i)
        return g

    @staticmethod
    def create_interval_graph(n):
        """
        Create interval graph with n vertices.
        Perfect graph: represents intervals on a line.
        """
        g = Graph()
        # Simple construction: each vertex i connected to vertices j where i < j ≤ i+2
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):
                g.add_edge(i, j)
        return g

    @staticmethod
    def create_chordal_graph(n):
        """
        Create chordal graph: a tree with additional edges.
        Perfect graph class.
        """
        g = Graph()
        # Create tree backbone
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        # Add chords to make it chordal
        for i in range(n - 2):
            g.add_edge(i, i + 2)
        return g

    @staticmethod
    def create_grid_graph(m, n):
        """Create m×n grid graph."""
        g = Graph()
        # Vertex (i,j) has label i*n + j
        for i in range(m):
            for j in range(n):
                v = i * n + j
                g.add_vertex(v)
                # Connect to right neighbor
                if j + 1 < n:
                    g.add_edge(v, i * n + (j + 1))
                # Connect to bottom neighbor
                if i + 1 < m:
                    g.add_edge(v, (i + 1) * n + j)
        return g

    @staticmethod
    def create_hypercube(d):
        """
        Create d-dimensional hypercube Qd.
        Each vertex is a binary string of length d.
        Two vertices are adjacent if they differ in exactly one bit.
        """
        g = Graph()
        n = 2 ** d
        
        # Vertices labeled 0 to 2^d - 1
        for i in range(n):
            g.add_vertex(i)
        
        # Add edges between vertices differing in one bit
        for i in range(n):
            for bit in range(d):
                j = i ^ (1 << bit)  # Flip bit
                if i < j:  # Add each edge once
                    g.add_edge(i, j)
        
        return g

    @staticmethod
    def create_complement_cycle(n):
        """Create complement of cycle Cn: removes all cycle edges."""
        g = Graph()
        for i in range(n):
            for j in range(i + 1, n):
                # Add edge if not in the cycle
                is_cycle_edge = (j == i + 1) or (i == 0 and j == n - 1)
                if not is_cycle_edge:
                    g.add_edge(i, j)
        return g

    @staticmethod
    def create_triangulation(n):
        """
        Create a random triangulated graph (chordal).
        Perfect graph: every cycle of length ≥ 4 has a chord.
        """
        g = Graph()
        # Start with K3
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        
        # Add vertices and connect to form triangles
        for i in range(3, n):
            # Connect to previous vertices forming triangles
            g.add_edge(i - 1, i)
            g.add_edge(i - 2, i)
            if i >= 3:
                g.add_edge(i - 3, i)
        
        return g

    @staticmethod
    def create_crown_graph(n):
        """
        Create crown graph: complete bipartite minus perfect matching.
        K_{n,n} with no edges between pairs (i,i).
        """
        g = Graph()
        for i in range(n):
            for j in range(n):
                if i != j:
                    g.add_edge(i, n + j)
        return g

    @staticmethod
    def create_kneser_graph(n, k):
        """
        Create Kneser graph K(n,k).
        Vertices: all k-subsets of {1,...,n}
        Edges: between disjoint subsets
        """
        from itertools import combinations
        
        g = Graph()
        subsets = list(combinations(range(n), k))
        
        for idx_i, subset_i in enumerate(subsets):
            for idx_j, subset_j in enumerate(subsets):
                if idx_i < idx_j:
                    # Check if disjoint
                    if len(set(subset_i) & set(subset_j)) == 0:
                        g.add_edge(idx_i, idx_j)
        
        return g

    @staticmethod
    def create_mycielski_graph(k):
        """
        Create Mycielski graph M_k.
        Triangle-free graphs with chromatic number k.
        Used to study perfect graphs.
        """
        if k == 1:
            g = Graph()
            g.add_vertex(0)
            return g
        if k == 2:
            g = Graph()
            g.add_edge(0, 1)
            return g
        
        # Recursive construction
        g = BenchmarkGraphs.create_mycielski_graph(k - 1)
        n = len(g)
        
        # Add new layer: vertices u'_i for each u_i
        for i in range(n):
            g.add_vertex(n + i)
        
        # Connect u'_i to all neighbors of u_i
        for i in range(n):
            for j in g.neighbors(i):
                g.add_edge(n + i, j)
        
        # Add central vertex connected to all u'_i
        central = 2 * n
        g.add_vertex(central)
        for i in range(n):
            g.add_edge(central, n + i)
        
        return g

    @staticmethod
    def create_random_regular(n, d):
        """
        Create random d-regular graph on n vertices.
        Approximate construction using configuration model.
        """
        if n * d % 2 != 0:
            raise ValueError("n*d must be even")
        
        g = Graph()
        for i in range(n):
            g.add_vertex(i)
        
        # Simple greedy approach to create d-regular graph
        degrees = [0] * n
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            u = 0
            while u < n and degrees[u] == d:
                u += 1
            if u == n:
                break  # All vertices have degree d
            
            # Find vertex with degree < d
            v = u + 1
            while v < n and degrees[v] == d:
                v += 1
            if v == n:
                break
            
            if not g.is_adjacent(u, v):
                g.add_edge(u, v)
                degrees[u] += 1
                degrees[v] += 1
            
            attempts += 1
        
        return g

    @staticmethod
    def create_random_connected(n, p):
        """
        Create random connected graph with edge probability p.
        Uses rejection sampling to ensure connectivity.
        """
        g = None
        while g is None or not g.is_connected():
            g = Graph()
            for i in range(n):
                g.add_vertex(i)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if (i + j) % 10 < p * 10:  # Pseudo-random
                        g.add_edge(i, j)
        
        return g

    @staticmethod
    def create_friendship_graph(n):
        """
        Create friendship graph: n triangles sharing a common vertex.
        χ(G) = 3, ω(G) = 3.
        """
        g = Graph()
        g.add_vertex(0)  # Central vertex
        
        for i in range(n):
            v1 = 1 + 2 * i
            v2 = 1 + 2 * i + 1
            g.add_edge(0, v1)
            g.add_edge(0, v2)
            g.add_edge(v1, v2)
        
        return g

    @staticmethod
    def create_paley_graph(q):
        """
        Create Paley graph of order q (where q ≡ 1 mod 4 is prime power).
        Vertices: elements of finite field F_q
        Edges: between quadratic residues
        
        Simplified version for demonstration.
        """
        if q <= 1 or q % 4 != 1:
            return BenchmarkGraphs.create_complete(q)
        
        g = Graph()
        for i in range(q):
            g.add_vertex(i)
        
        # Simplified: connect vertices whose difference is a quadratic residue
        residues = set()
        for i in range(q):
            residues.add((i * i) % q)
        
        for i in range(q):
            for j in range(i + 1, q):
                if ((i - j) % q) in residues:
                    g.add_edge(i, j)
        
        return g

    @staticmethod
    def create_strongly_regular(n, k, lambda_param, mu_param):
        """
        Create strongly regular graph srg(n, k, λ, μ).
        n vertices, k-regular, λ common neighbors for adjacent pairs,
        μ common neighbors for non-adjacent pairs.
        
        Approximate construction.
        """
        g = Graph()
        for i in range(n):
            g.add_vertex(i)
        
        # Simplified: Create regular graph structure
        for i in range(n):
            for j in range(i + 1, min(i + k + 1, n)):
                if j - i <= k:
                    g.add_edge(i, j)
        
        return g

    @staticmethod
    def get_all_benchmarks():
        """
        Return dictionary of all benchmark graphs for testing.
        Key: graph name, Value: (factory_function, expected_omega, expected_chromatic)
        """
        benchmarks = {
            "Pentagon (C5)": (
                BenchmarkGraphs.create_pentagon,
                5,
                3,
                "Classic odd cycle, θ=√5 from Paper 1"
            ),
            "Petersen": (
                BenchmarkGraphs.create_petersen,
                10,
                3,
                "Paper 1 reference: ω=2, χ=3, θ=4"
            ),
            "C4 (4-cycle)": (
                lambda: BenchmarkGraphs.create_cycle(4),
                4,
                2,
                "Even cycle, bipartite, θ=2"
            ),
            "C6 (6-cycle)": (
                lambda: BenchmarkGraphs.create_cycle(6),
                6,
                2,
                "Bipartite cycle"
            ),
            "C7 (7-cycle)": (
                lambda: BenchmarkGraphs.create_cycle(7),
                7,
                3,
                "Odd cycle, χ=3"
            ),
            "K3 (Triangle)": (
                lambda: BenchmarkGraphs.create_complete(3),
                3,
                3,
                "Clique: ω=χ=3"
            ),
            "K4": (
                lambda: BenchmarkGraphs.create_complete(4),
                4,
                4,
                "Clique: ω=χ=4"
            ),
            "K5": (
                lambda: BenchmarkGraphs.create_complete(5),
                5,
                5,
                "Clique: ω=χ=5"
            ),
            "K6": (
                lambda: BenchmarkGraphs.create_complete(6),
                6,
                6,
                "Clique: ω=χ=6"
            ),
            "K(3,3)": (
                lambda: BenchmarkGraphs.create_complete_bipartite(3, 3),
                6,
                2,
                "Complete bipartite, perfect"
            ),
            "K(4,4)": (
                lambda: BenchmarkGraphs.create_complete_bipartite(4, 4),
                8,
                2,
                "Complete bipartite"
            ),
            "K(5,5)": (
                lambda: BenchmarkGraphs.create_complete_bipartite(5, 5),
                10,
                2,
                "Complete bipartite"
            ),
            "Interval G(5)": (
                lambda: BenchmarkGraphs.create_interval_graph(5),
                5,
                2,
                "Perfect graph class"
            ),
            "Interval G(8)": (
                lambda: BenchmarkGraphs.create_interval_graph(8),
                8,
                2,
                "Interval graph"
            ),
            "Chordal G(6)": (
                lambda: BenchmarkGraphs.create_chordal_graph(6),
                6,
                3,
                "Perfect: chordal class"
            ),
            "Chordal G(8)": (
                lambda: BenchmarkGraphs.create_chordal_graph(8),
                8,
                3,
                "Chordal graph"
            ),
            "Grid 3×3": (
                lambda: BenchmarkGraphs.create_grid_graph(3, 3),
                9,
                2,
                "Bipartite, χ=2"
            ),
            "Grid 4×4": (
                lambda: BenchmarkGraphs.create_grid_graph(4, 4),
                16,
                2,
                "Bipartite grid"
            ),
            "Hypercube Q3": (
                lambda: BenchmarkGraphs.create_hypercube(3),
                8,
                2,
                "Bipartite, χ=2"
            ),
            "Hypercube Q4": (
                lambda: BenchmarkGraphs.create_hypercube(4),
                16,
                2,
                "Bipartite"
            ),
            "Star S(8)": (
                lambda: BenchmarkGraphs.create_star(8),
                8,
                2,
                "Bipartite, χ=2"
            ),
            "Path P(10)": (
                lambda: BenchmarkGraphs.create_path(10),
                10,
                2,
                "Bipartite path, χ=2"
            ),
            "Wheel W(6)": (
                lambda: BenchmarkGraphs.create_wheel(6),
                6,
                3,
                "Wheel graph"
            ),
            "Triangulation T(8)": (
                lambda: BenchmarkGraphs.create_triangulation(8),
                8,
                2,
                "Chordal, perfect"
            ),
            "Crown G(4)": (
                lambda: BenchmarkGraphs.create_crown_graph(4),
                8,
                2,
                "K_{n,n} minus matching"
            ),
            "Friendship F(5)": (
                lambda: BenchmarkGraphs.create_friendship_graph(5),
                11,
                3,
                "Friendship graph"
            ),
            "Complement C5": (
                lambda: BenchmarkGraphs.create_complement_cycle(5),
                5,
                3,
                "Complement of pentagon"
            ),
        }
        
        return benchmarks
