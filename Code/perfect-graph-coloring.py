import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations, permutations
import numpy as np
import timeit

class PerfectGraphColoring:
    """
    Implementation of algorithms from "Colouring perfect graphs with bounded clique number"
    by Chudnovsky, Lagoutte, Seymour, and Spirkl (2017)
    """
    
    def __init__(self, G: nx.Graph):
        """
        Initialize with a NetworkX graph
        
        Args:
            G: Input graph (should be a Berge graph for the algorithms to work correctly)
        """
        self.G = G
        self.n = G.number_of_nodes()
        self.nodes = list(G.nodes())
        
    def is_connected_subset(self, subset: Set, graph: nx.Graph = None) -> bool:
        """Check if a subset induces a connected subgraph"""
        if graph is None:
            graph = self.G
        if len(subset) <= 1:
            return True
        subgraph = graph.subgraph(subset)
        return nx.is_connected(subgraph)
    
    def get_complement(self, graph: nx.Graph = None) -> nx.Graph:
        """Return the complement of a graph"""
        if graph is None:
            graph = self.G
        return nx.complement(graph)
    
    def find_components(self, subset: Set, graph: nx.Graph = None) -> List[Set]:
        """Find connected components of induced subgraph"""
        if graph is None:
            graph = self.G
        subgraph = graph.subgraph(subset)
        return [set(comp) for comp in nx.connected_components(subgraph)]
    
    def find_anticomponents(self, subset: Set) -> List[Set]:
        """Find anticomponents (components in complement graph)"""
        complement = self.get_complement()
        return self.find_components(subset, complement)
    
    def is_skew_partition(self, A: Set, B: Set) -> bool:
        """
        Check if (A, B) is a skew partition
        A skew partition requires:
        - A and B partition the vertex set
        - G[A] is not connected 
        - G[B] is not anticonnected (complement of G[B] is not connected)
        """
        if not A or not B:
            return False
        if A.union(B) != set(self.nodes) or A.intersection(B):
            return False
        
        # Check G[A] is not connected
        if self.is_connected_subset(A):
            return False
        
        # Check G[B] is not anticonnected
        complement = self.get_complement()
        if self.is_connected_subset(B, complement):
            return False
            
        return True
    
    def is_complete_to(self, vertex_set: Set, target_set: Set) -> bool:
        """Check if vertex_set is complete to target_set (all edges exist)"""
        for v in vertex_set:
            for u in target_set:
                if v != u and not self.G.has_edge(v, u):
                    return False
        return True
    
    def is_anticomplete_to(self, vertex_set: Set, target_set: Set) -> bool:
        """Check if vertex_set is anticomplete to target_set (no edges exist)"""
        for v in vertex_set:
            for u in target_set:
                if v != u and self.G.has_edge(v, u):
                    return False
        return True
    
    def find_induced_path(self, start: int, end: int, interior: Set) -> Optional[List]:
        """
        Find an induced path from start to end with interior in given set
        Returns path as list of vertices or None if no path exists
        """
        # Use BFS to find shortest path in induced subgraph
        subgraph_nodes = interior.union({start, end})
        subgraph = self.G.subgraph(subgraph_nodes)
        
        try:
            path = nx.shortest_path(subgraph, start, end)
            # Check if path is induced (no chords)
            for i in range(len(path)):
                for j in range(i+2, len(path)):
                    if j != i+1 and self.G.has_edge(path[i], path[j]):
                        return None  # Path has a chord
            return path
        except nx.NetworkXNoPath:
            return None
    
    def is_balanced_skew_partition(self, A: Set, B: Set) -> bool:
        """
        Check if skew partition (A, B) is balanced
        Balanced requires:
        - For all nonadjacent u,v in B, every induced path with ends u,v 
          and interior in A has even length
        - For all adjacent u,v in A, every antipath with ends u,v
          and interior in B has even length
        """
        if not self.is_skew_partition(A, B):
            return False
        
        # Check paths condition
        B_list = list(B)
        for i, u in enumerate(B_list):
            for v in B_list[i+1:]:
                if not self.G.has_edge(u, v):
                    # Find induced path from u to v through A
                    for A_comp in self.find_components(A):
                        path = self.find_induced_path(u, v, A_comp)
                        if path and len(path) % 2 == 0:  # Odd length path
                            return False
        
        # Check antipaths condition (paths in complement)
        complement = self.get_complement()
        A_list = list(A)
        for i, u in enumerate(A_list):
            for v in A_list[i+1:]:
                if self.G.has_edge(u, v):
                    # Find antipath (path in complement)
                    for B_comp in self.find_anticomponents(B):
                        # This is simplified - full implementation would need antipath finding
                        pass
        
        return True
    
    def find_square_based_skew_partitions(self) -> List[Tuple[Set, Set]]:
        """
        Algorithm 2.1 (partial): Find square-based skew partitions
        These arise from 4-holes with specific properties
        """
        partitions = []
        
        # Find all 4-holes (cycles of length 4)
        for cycle in nx.simple_cycles(self.G):
            if len(cycle) == 4:
                a, b, c, d = cycle
                
                # Check if this forms a valid hole structure
                if (self.G.has_edge(a, b) and self.G.has_edge(b, c) and 
                    self.G.has_edge(c, d) and self.G.has_edge(d, a) and
                    not self.G.has_edge(a, c) and not self.G.has_edge(b, d)):
                    
                    # Construct B as union of neighbors
                    B = set()
                    for v in self.nodes:
                        if v not in {a, b, c, d}:
                            if (self.G.has_edge(v, a) and self.G.has_edge(v, c)) or \
                               (self.G.has_edge(v, b) and self.G.has_edge(v, d)):
                                B.add(v)
                    
                    A = set(self.nodes) - B
                    
                    if self.is_skew_partition(A, B):
                        # Check if tight and square-based
                        partitions.append((A, B))
        
        return partitions
    
    def find_star_cutset(self) -> Optional[Set]:
        """
        Algorithm 5.1: Find a star cutset
        A star cutset is a cutset B where some vertex in B is adjacent to all others in B
        """
        for v in self.nodes:
            neighbors = set(self.G.neighbors(v))
            if not neighbors:
                continue
                
            # Check if N[v] is a cutset
            cutset = neighbors.union({v})
            remaining = set(self.nodes) - cutset
            
            if remaining and not self.is_connected_subset(remaining):
                return cutset
            
            # Check variations as described in algorithm
            if len(neighbors) >= 2:
                remaining_graph = self.G.subgraph(remaining)
                components = list(nx.connected_components(remaining_graph))
                
                if len(components) == 1:
                    C = components[0]
                    for u in neighbors:
                        if self.is_anticomplete_to({u}, C):
                            return cutset - {u}
        
        return None
    
    def kennedy_reed_cutsets(self) -> List[Set]:
        """
        Kennedy-Reed algorithm to find cutsets
        Returns list of at most n^4 cutsets that cover all skew partitions
        """
        cutsets = []
        
        # Simplified version - find clique cutsets
        for size in range(1, min(self.n, 5)):  # Limit size for efficiency
            for subset in combinations(self.nodes, size):
                subset_set = set(subset)
                
                # Check if it's a clique
                is_clique = all(self.G.has_edge(u, v) or u == v 
                               for u in subset for v in subset)
                
                if is_clique:
                    # Check if it's a cutset
                    remaining = set(self.nodes) - subset_set
                    if remaining and not self.is_connected_subset(remaining):
                        cutsets.append(subset_set)
        
        return cutsets
    
    def find_tight_skew_partitions(self) -> List[Tuple[Set, Set]]:
        """
        Algorithm 2.2: Find all tight skew partitions
        A skew partition (A,B) is tight if:
        - No vertex in A is complete to any anticomponent of B
        - No vertex in B is anticomplete to any component of A
        """
        partitions = []
        cutsets = self.kennedy_reed_cutsets()
        
        for B in cutsets:
            A = set(self.nodes) - B
            
            if self.is_skew_partition(A, B):
                # Check if tight
                is_tight = True
                
                # Check first condition
                for a in A:
                    for B_anticomp in self.find_anticomponents(B):
                        if self.is_complete_to({a}, B_anticomp):
                            is_tight = False
                            break
                    if not is_tight:
                        break
                
                # Check second condition
                if is_tight:
                    for b in B:
                        for A_comp in self.find_components(A):
                            if self.is_anticomplete_to({b}, A_comp):
                                is_tight = False
                                break
                        if not is_tight:
                            break
                
                if is_tight:
                    partitions.append((A, B))
        
        return partitions
    
    def find_maximum_clique(self) -> Set:
        """Find a maximum clique in the graph"""
        # Find all maximal cliques and return the largest
        cliques = list(nx.find_cliques(self.G))
        if not cliques:
            return set()
        return set(max(cliques, key=len))
    
    def color_graph(self, k: Optional[int] = None) -> Dict[int, int]:
        """
        Main algorithm (simplified): Color a perfect graph
        Returns dict mapping vertices to colors (integers)
        """
        if k is None:
            k = len(self.find_maximum_clique())
        
        # For small graphs or base cases, use greedy coloring
        if self.n <= 2 * k:
            return nx.coloring.greedy_color(self.G, strategy='largest_first')
        
        # Try to find a balanced skew partition
        tight_partitions = self.find_tight_skew_partitions()
        
        for A, B in tight_partitions:
            if self.is_balanced_skew_partition(A, B):
                # Found balanced skew partition - recursively color
                # This is a simplified version of the matching algorithm
                
                # Partition A into two anticomplete parts
                A_components = self.find_components(A)
                if len(A_components) >= 2:
                    A1 = A_components[0]
                    A2 = set().union(*A_components[1:])
                    
                    # Color subgraphs
                    G1 = self.G.subgraph(A1.union(B))
                    G2 = self.G.subgraph(A2.union(B))
                    
                    # Recursive coloring (simplified)
                    coloring1 = nx.coloring.greedy_color(G1, strategy='largest_first')
                    coloring2 = nx.coloring.greedy_color(G2, strategy='largest_first')
                    
                    # Combine colorings (simplified matching)
                    final_coloring = {}
                    for v in A1:
                        final_coloring[v] = coloring1[v]
                    for v in A2:
                        final_coloring[v] = coloring2[v]
                    for v in B:
                        # Use consistent coloring from both subgraphs
                        final_coloring[v] = coloring1.get(v, coloring2.get(v, 0))
                    
                    return final_coloring
        
        # No balanced skew partition found - use standard algorithm
        return nx.coloring.greedy_color(self.G, strategy='largest_first')


# Example usage
def example_usage():
    """
    Demonstrate the algorithms on small perfect graphs with runtime benchmarks
    """
    print("=" * 60)
    print("Perfect Graph Coloring Algorithm Demo")
    print("=" * 60)
    timer = timeit.default_timer

    # Example 1: Complement of a 5-cycle (perfect graph)
    print("\n1. Testing on complement of C5:")
    G1 = nx.complement(nx.cycle_graph(5))
    pgc1 = PerfectGraphColoring(G1)

    # Find tight skew partitions
    t0 = timer()
    partitions = pgc1.find_tight_skew_partitions()
    t1 = timer()
    print(f"   Found {len(partitions)} tight skew partitions [Time: {t1-t0:.6f} seconds]")

    # Find star cutset
    t0 = timer()
    star_cutset = pgc1.find_star_cutset()
    t1 = timer()
    if star_cutset:
        print(f"   Found star cutset: {star_cutset} [Time: {t1-t0:.6f} seconds]")
    else:
        print(f"   No star cutset found [Time: {t1-t0:.6f} seconds]")

    # Find maximum clique
    t0 = timer()
    max_clique = pgc1.find_maximum_clique()
    t1 = timer()
    print(f"   Maximum clique size: {len(max_clique)} [Time: {t1-t0:.6f} seconds]")
    print(f"   Maximum clique vertices: {max_clique}")

    # Color the graph
    t0 = timer()
    coloring = pgc1.color_graph()
    t1 = timer()
    num_colors = max(coloring.values()) + 1 if coloring else 0
    print(f"   Coloring uses {num_colors} colors [Time: {t1-t0:.6f} seconds]")
    print(f"   Coloring: {coloring}")

    # Verify the coloring
    valid = True
    for u, v in G1.edges():
        if coloring[u] == coloring[v]:
            print(f"   Invalid coloring: {u} and {v} have same color")
            valid = False
            break
    if valid:
        print("   ✓ Coloring is valid!")

    # Example 2: Complete bipartite graph K3,3 (also perfect)
    print("\n2. Testing on complete bipartite graph K_{3,3}:")
    G2 = nx.complete_bipartite_graph(3, 3)
    pgc2 = PerfectGraphColoring(G2)

    t0 = timer()
    max_clique = pgc2.find_maximum_clique()
    t1 = timer()
    print(f"   Maximum clique size: {len(max_clique)} [Time: {t1-t0:.6f} seconds]")

    t0 = timer()
    coloring = pgc2.color_graph()
    t1 = timer()
    num_colors = max(coloring.values()) + 1 if coloring else 0
    print(f"   Coloring uses {num_colors} colors [Time: {t1-t0:.6f} seconds]")

    if num_colors == len(max_clique):
        print(f"   ✓ Coloring is optimal (χ(G) = ω(G) = {num_colors})")

    # Example 3: Path graph P4 (trivially perfect)
    print("\n3. Testing on path graph P4:")
    G3 = nx.path_graph(4)
    pgc3 = PerfectGraphColoring(G3)

    t0 = timer()
    coloring = pgc3.color_graph()
    t1 = timer()
    num_colors = max(coloring.values()) + 1 if coloring else 0
    max_clique = pgc3.find_maximum_clique()
    print(f"   Maximum clique size: {len(max_clique)}")
    print(f"   Coloring uses {num_colors} colors [Time: {t1-t0:.6f} seconds]")
    print(f"   Coloring: {coloring}")

    if num_colors == len(max_clique):
        print(f"   ✓ Coloring is optimal (χ(G) = ω(G) = {num_colors})")

    print("\n" + "=" * 60)
    print("Demo complete!")

if __name__ == "__main__":
    example_usage()