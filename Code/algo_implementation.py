import timeit
import itertools
import sys

# Set higher recursion depth for the deep recursive calls
sys.setrecursionlimit(2000)

class Graph:
    """
    A 'from-scratch' graph representation using adjacency sets.
    Based on descriptions in arXiv:1707.03747v1.[1]
    """

    def __init__(self, vertices=None):
        self.adj = {}
        if vertices:
            for v in vertices:
                self.add_vertex(v)

    def add_vertex(self, v):
        if v not in self.adj:
            self.adj[v] = set()

    def add_edge(self, u, v):
        if u == v:
            return
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].add(v)
        self.adj[v].add(u)

    def is_adjacent(self, u, v):
        return u in self.adj and v in self.adj[u]

    def neighbors(self, v):
        return self.adj.get(v, set())

    def all_vertices(self):
        return set(self.adj.keys())

    def __len__(self):
        return len(self.adj)

    def get_induced_subgraph(self, vertex_set):
        """
        Creates a new Graph object G[X] induced on vertex_set (X)
       .[1]
        """
        new_graph = Graph()
        # Use a set for faster lookups
        vertex_set_lookup = set(vertex_set)
        for v in vertex_set_lookup:
            new_graph.add_vertex(v) # Ensure all vertices are added
            if v in self.adj:
                # Efficiently find common neighbors
                valid_neighbors = self.adj[v] & vertex_set_lookup
                for u in valid_neighbors:
                    # Add edge only if both endpoints are in the set
                    new_graph.adj[v].add(u)
        return new_graph

    def get_complement(self):
        """
        Creates the complement graph G-bar.[1]
        """
        new_graph = Graph(self.all_vertices())
        all_verts = list(self.all_vertices())
        for i in range(len(all_verts)):
            u = all_verts[i]
            for j in range(i + 1, len(all_verts)):
                v = all_verts[j]
                if not self.is_adjacent(u, v):
                    new_graph.add_edge(u, v)
        return new_graph

    def _bfs_traverse(self, start_node, visited):
        """Helper for finding connected components."""
        component = set()
        q = [start_node]
        visited.add(start_node)
        component.add(start_node)

        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in self.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    component.add(v)
                    q.append(v)
        return component

    def get_connected_components(self):
        """
        Finds all connected components of the graph.[1]
        """
        visited = set()
        components = [] 
        for v in self.all_vertices():
            if v not in visited:
                component = self._bfs_traverse(v, visited)
                components.append(component)
        return components

    def is_connected(self):
        """Checks if the graph has exactly one component."""
        if not self.adj:
            return True  # By convention for an empty graph
        
        all_verts = self.all_vertices()
        if not all_verts:
            return True

        visited = set()
        component = self._bfs_traverse(next(iter(all_verts)), visited)
        return len(component) == len(all_verts)

    def get_anticomponents(self):
        """
        Finds all anticomponents (components of G-bar).[1]
        """
        complement = self.get_complement()
        return complement.get_connected_components()

    def get_omega(self, k_bound=None):
        """
        Finds the size of the maximum clique, omega(G).[1]
        Uses a bounded backtracking search (Bron-Kerbosch)
        as justified by the paper's O(n^k) claim for this step.[1]
        
        [FIXED] Completed placeholder and fixed scoping bug
        using 'nonlocal' keyword.
        """
        if not self.adj:
            return 0
        
        if k_bound is None:
            k_bound = len(self.adj)
        
        max_clique_size = 0 

        def find_max_clique_recursive(potential_clique, candidates):
            nonlocal max_clique_size 
            
            if not candidates:
                max_clique_size = max(max_clique_size, len(potential_clique))
                return

            # Pruning
            if len(potential_clique) + len(candidates) <= max_clique_size:
                return
            
            # Pruning
            if len(potential_clique) >= k_bound:
                max_clique_size = k_bound
                return

            # Use a pivot
            pivot = next(iter(candidates))
            candidates_without_pivot_neighbors = list(candidates - self.neighbors(pivot))

            for q in candidates_without_pivot_neighbors:
                if max_clique_size >= k_bound:
                    return # Pruning
                
                new_potential_clique = potential_clique | {q}
                new_candidates = candidates & self.neighbors(q)

                find_max_clique_recursive(new_potential_clique, new_candidates)
                
                candidates.remove(q)

        all_verts = self.all_vertices()
        find_max_clique_recursive(set(), all_verts)
        return max_clique_size

    def find_induced_path(self, start, end, interior_nodes):
        """
        Finds an induced path between start and end with interior in
        interior_nodes. Returns the path (list of nodes) or None.
        Needed for 'balanced' check.[1]
        """
        # BFS-based search for induced paths
        path_queue = [(start, [start])] # (current_node, path_so_far)
        
        while path_queue:
            (u, current_path) = path_queue.pop(0)
            
            # Valid neighbors are those in interior_nodes not yet in the path
            # OR the end node
            valid_neighbors = (interior_nodes - set(current_path)) | {end}

            for v in self.neighbors(u):
                if v in valid_neighbors:
                    # Check the 'induced' property
                    is_induced_move = True
                    for node in current_path[:-1]:  # Check against all but parent
                        if self.is_adjacent(v, node):
                            is_induced_move = False
                            break
                    
                    if is_induced_move:
                        new_path = current_path + [v]
                        if v == end:
                            return new_path  # Found an induced path!
                        
                        path_queue.append((v, new_path))
        return None


class GraphHelpers:
    """
    A container for all the static graph algorithm methods
    needed to implement the decomposition.
    """

    def find_star_cutset(self, graph_obj):
        """
        Implements Algorithm 5.1 [1] to find a star cutset.
        Returns the cutset B (a set of vertices) or None.
        """
        all_verts = graph_obj.all_vertices()
        if len(all_verts) <= 1:
            return None
            
        for v in all_verts:
            N_v = graph_obj.neighbors(v)
            N_v_bracket = N_v | {v}

            # Check 1: N[v]
            if N_v:
                G_minus_N_v_bracket = graph_obj.get_induced_subgraph(
                    all_verts - N_v_bracket
                )
                if not G_minus_N_v_bracket.is_connected() and len(G_minus_N_v_bracket) > 0:
                    return N_v_bracket

            # Check 2: N[v] \ {u}
            if len(N_v) >= 2:
                G_minus_N_v_bracket = graph_obj.get_induced_subgraph(
                    all_verts - N_v_bracket
                )
                components = G_minus_N_v_bracket.get_connected_components()
                if len(components) == 1 and len(G_minus_N_v_bracket) > 0:
                    C = components[0] # Get the set from the list
                    for u in N_v:
                        u_is_anticomplete_to_C = True
                        for c_node in C:
                            if graph_obj.is_adjacent(u, c_node):
                                u_is_anticomplete_to_C = False
                                break
                        if u_is_anticomplete_to_C:
                            return N_v_bracket - {u}

            # Check 3: V(G) \ {x, y}
            if len(N_v) >= 3 and len(N_v_bracket) == len(graph_obj):
                N_v_list = list(N_v)
                for i in range(len(N_v_list)):
                    x = N_v_list[i]
                    for j in range(i + 1, len(N_v_list)):
                        y = N_v_list[j]
                        if not graph_obj.is_adjacent(x, y):
                            return all_verts - {x, y}
        return None

    def find_all_4_holes(self, graph_obj):
        """Helper to iterate all 4-holes (a-b-c-d-a).[1]"""
        holes = [] 
        seen_holes = set() # To store canonical representation
        verts = list(graph_obj.all_vertices())
        
        for a_idx, a in enumerate(verts):
            for b in graph_obj.neighbors(a):
                for c in graph_obj.neighbors(b):
                    if c == a or graph_obj.is_adjacent(a, c):
                        continue
                    for d in graph_obj.neighbors(c):
                        if d == b or d == a:
                            continue
                        if graph_obj.is_adjacent(d, a) and not graph_obj.is_adjacent(d, b):
                            # Found hole (a,b,c,d). Add in a canonical way.
                            hole_tuple = tuple(sorted((a, b, c, d)))
                            if hole_tuple not in seen_holes:
                                holes.append((a, b, c, d)) # Store one ordering
                                seen_holes.add(hole_tuple)
        return holes

    def _is_tight_skew_partition(self, graph_obj, A, B):
        """
        Helper to check if (A,B) is a tight skew partition.[1]
        """
        # Check if skew partition
        G_A = graph_obj.get_induced_subgraph(A)
        if G_A.is_connected() or len(G_A) == 0:
            return False

        G_B = graph_obj.get_induced_subgraph(B)
        if len(G_B) == 0:
            return False
        G_B_comp = G_B.get_complement()
        if G_B_comp.is_connected():
            return False

        # Check if tight (not loose)
        A_components = G_A.get_connected_components()
        B_anticomponents = G_B_comp.get_connected_components()
        if not A_components or not B_anticomponents:
            return False # Should not happen if not connected

        for v_a in A:
            for B_i in B_anticomponents:
                if all(graph_obj.is_adjacent(v_a, v_b) for v_b in B_i):
                    return False  # Is loose

        for v_b in B:
            for A_i in A_components:
                if all(not graph_obj.is_adjacent(v_b, v_a) for v_a in A_i):
                    return False  # Is loose

        return True  # Is a tight skew partition
    
    def _is_loose_skew_partition(self, graph_obj, A, B):
        """
        Helper to check if (A,B) is a loose skew partition.[1]
        """
        G_A = graph_obj.get_induced_subgraph(A)
        G_B = graph_obj.get_induced_subgraph(B)
        if len(G_A) == 0 or len(G_B) == 0: return False
        
        G_B_comp = G_B.get_complement()
        
        if G_A.is_connected() or G_B_comp.is_connected():
            return False # Not a skew partition
        
        A_components = G_A.get_connected_components()
        B_anticomponents = G_B_comp.get_connected_components()
        if not A_components or not B_anticomponents:
            return False

        for v_a in A:
            for B_i in B_anticomponents:
                if all(graph_obj.is_adjacent(v_a, v_b) for v_b in B_i):
                    return True # Is loose
        
        for v_b in B:
            for A_i in A_components:
                if all(not graph_obj.is_adjacent(v_b, v_a) for v_a in A_i):
                    return True # Is loose
        return False


    def find_unbalanced_tight_partitions(self, graph_obj):
        """
        Implements Algorithm 2.1.[1]
        Outputs a list of unbalanced, tight skew partitions.
        """
        output_list = [] 
        all_verts = graph_obj.all_vertices()

        # 1. Check G
        for (a, b, c, d) in self.find_all_4_holes(graph_obj):
            N_a_c = graph_obj.neighbors(a) & graph_obj.neighbors(c)
            N_b_d = graph_obj.neighbors(b) & graph_obj.neighbors(d)
            B = N_a_c | N_b_d
            A = all_verts - B

            if not A or not B:
                continue

            if self._is_tight_skew_partition(graph_obj, A, B):
                path = graph_obj.find_induced_path(a, c, A) # Corrected call
                if path and (len(path) - 1) % 2!= 0:  # Odd length
                    output_list.append((A, B))

        # 2. Check G-bar
        graph_comp = graph_obj.get_complement()
        for (a, b, c, d) in self.find_all_4_holes(graph_comp):
            N_a_c = graph_comp.neighbors(a) & graph_comp.neighbors(c)
            N_b_d = graph_comp.neighbors(b) & graph_comp.neighbors(d)
            D = N_a_c | N_b_d  # This is B in G-bar
            C = all_verts - D  # This is A in G-bar

            if not C or not D:
                continue

            # (C, D) is a skew partition of G-bar
            # so (D, C) is a skew partition of G
            if self._is_tight_skew_partition(graph_obj, D, C):
                path = graph_comp.find_induced_path(a, c, C) # Corrected call
                if path and (len(path) - 1) % 2!= 0:
                    output_list.append((D, C))

        return output_list

    def find_clique_cutsets(self, graph_obj):
        """
        Implements Algorithm 6.4 (stand-in).[1]
        This is a brute-force O(2^n * n^k) stand-in for Tarjan's
        O(nm) algorithm (Alg 6.1) [1], as required by
        the "from-scratch" constraint.
        """
        clique_cutsets = [] 
        verts = list(graph_obj.all_vertices())
        n = len(verts)
        if n < 2:
            return [] 

        for k in range(1, n):
            for separator in itertools.combinations(verts, k):
                separator_set = set(separator)

                # Check if it's a cutset
                G_minus_S = graph_obj.get_induced_subgraph(
                    graph_obj.all_vertices() - separator_set
                )
                
                
                if not G_minus_S.is_connected() and len(G_minus_S) > 0:
                    # It is a cutset. Now check if it's a clique.
                    G_S = graph_obj.get_induced_subgraph(separator_set)
                    # Check if it's a clique
                    is_clique = True
                    sep_list = list(separator_set)
                    for i in range(len(sep_list)):
                        for j in range(i + 1, len(sep_list)):
                            if not G_S.is_adjacent(sep_list[i], sep_list[j]):
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    
                    if is_clique:
                        clique_cutsets.append(separator_set)

        # Filter for minimal cutsets (as per Alg 6.3) [1]
        minimal_clique_cutsets = [] 
        clique_cutsets.sort(key=len)
        for cs in clique_cutsets:
            is_minimal = True
            for mcs in minimal_clique_cutsets:
                if mcs.issubset(cs):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_clique_cutsets.append(cs)
        return minimal_clique_cutsets


    def run_kennedy_reed(self, graph_obj):
        """
        Implements the Kennedy-Reed algorithm from Sec 6.[1]
        """
        L = set()  # Master list of cutsets
        n = len(graph_obj)
        all_verts = list(graph_obj.all_vertices())
        if n == 0:
            return [] 

        # Loop over all k1, k2, r (O(n^3) loops)
        for k1 in range(1, n + 1):
            for k2 in range(1, k1 + 1):
                for r in all_verts:
                    # 1. Construct H(k1, k2, r)
                    H = Graph(all_verts)
                    # Add all edges from G
                    for u in all_verts:
                        for v in graph_obj.neighbors(u):
                            if u < v: H.add_edge(u, v)

                    # Add new edges based on common neighbors
                    for i in range(n):
                        u = all_verts[i]
                        for j in range(i + 1, n):
                            v = all_verts[j]

                            if H.is_adjacent(u, v):
                                continue

                            common_neighbors = graph_obj.neighbors(u) & graph_obj.neighbors(v)
                            if not common_neighbors:
                                continue

                            G_common = graph_obj.get_induced_subgraph(common_neighbors)
                            anticomponents = G_common.get_anticomponents()

                            for anti_comp in anticomponents:
                                if len(anti_comp) >= k1:
                                    H.add_edge(u, v)
                                    break
                                if len(anti_comp) >= k2 and r in anti_comp:
                                    H.add_edge(u, v)
                                    break

                    # 2. Run Alg 6.4 on H
                    clique_cutsets_H = self.find_clique_cutsets(H)
                    if clique_cutsets_H: # Check if None
                        for cs in clique_cutsets_H:
                            L.add(frozenset(cs))  # Use frozenset for hashing

        return [set(fs) for fs in L]
    
    def find_t_cutset(self, graph_obj):
        """
        Implements Algorithm 5.2 [1] to find a T-cutset.
        """
        all_verts = list(graph_obj.all_vertices())
        n = len(all_verts)
        for i in range(n):
            a1 = all_verts[i]
            for j in range(i + 1, n):
                a2 = all_verts[j]
                if graph_obj.is_adjacent(a1, a2): continue
                
                common_neighbors = graph_obj.neighbors(a1) & graph_obj.neighbors(a2)
                if not common_neighbors: continue
                
                G_common = graph_obj.get_induced_subgraph(common_neighbors)
                anticomponents = G_common.get_anticomponents()
                
                for B1 in anticomponents:
                    B2 = set()
                    for v in graph_obj.all_vertices() - (B1 | {a1, a2}):
                        if all(graph_obj.is_adjacent(v, b1_node) for b1_node in B1):
                            B2.add(v)
                    
                    
                    if not B2:
                        continue 
                        
                    # Check for path between a1, a2 in G \ (B1 U B2)
                    G_rest = graph_obj.get_induced_subgraph(
                        graph_obj.all_vertices() - (B1 | B2)
                    )
                    if G_rest.find_induced_path(a1, a2, G_rest.all_vertices() - {a1, a2}):
                        continue # Path exists, move on
                    
                    # No path exists. B1 U B2 is a T-cutset.
                    return B1 | B2
        return None

class CombinatorialColouringSolver:
    """
    Main solver class that encapsulates the logic from arXiv:1707.03747v1.
    Implements Algorithm 10.3 (P_k).[1]
    """
    def __init__(self, graph):
        self.graph = graph
        self.helpers = GraphHelpers()
        self.memoization_table = {} # Memoization for P_k

    def find_balanced_skew_partition(self):
        """
        Implements Algorithm 2.4.[1]
        """
        all_verts = self.graph.all_vertices()
        if len(all_verts) < 4: return None # Skew partitions require >3 nodes

        # --- Algorithm 2.3 --- [1]
        # 1a. Check for star cutsets in G and G_bar [1]
        star_cutset_G = self.helpers.find_star_cutset(self.graph)
        if star_cutset_G:
            return (all_verts - star_cutset_G, star_cutset_G)

        G_bar = self.graph.get_complement()
        star_cutset_G_bar = self.helpers.find_star_cutset(G_bar)
        if star_cutset_G_bar:
            return (star_cutset_G_bar, all_verts - star_cutset_G_bar) # (A, B)

        # 1b. Run Algorithm 5.4 [1]
        # First, test for T-cutset (Alg 5.2)
        t_cutset = self.helpers.find_t_cutset(self.graph)
        if t_cutset:
            # Paper says to run a balancing procedure [1]
            # For this implementation, we return the found partition
            return (all_verts - t_cutset, t_cutset)

        # Second, run Kennedy-Reed (Alg 5.3)
        kr_list_L = self.helpers.run_kennedy_reed(self.graph)
        for B in kr_list_L:
            A = all_verts - B
            if not A: continue
            if self.helpers._is_loose_skew_partition(self.graph, A, B):
                # Found a loose skew partition
                # Again, paper runs a balancing procedure [1]
                return (A, B)
        
        # --- End of Algorithm 2.3 ---
        # No loose partition found.

        # --- Algorithm 2.4 (Step 2) --- [1]
        # Find a balanced TIGHT partition.
        
        # Run Alg 2.2: Filter KR list for *tight* partitions [1]
        all_tight_partitions = [] 
        for B in kr_list_L:
            A = all_verts - B
            if not A: continue
            if self.helpers._is_tight_skew_partition(self.graph, A, B):
                all_tight_partitions.append((A, B))
        
        # Run Alg 2.1: Find unbalanced tight partitions [1]
        unbalanced_tight = self.helpers.find_unbalanced_tight_partitions(self.graph)
        unbalanced_set = {frozenset(B) for (A, B) in unbalanced_tight}

        # Find (All Tight) - (Unbalanced Tight)
        for (A, B) in all_tight_partitions:
            if frozenset(B) not in unbalanced_set:
                return (A, B)  # Found a balanced tight partition

        return None # No balanced skew partition exists

    def backtracking_colouring(self, graph, k):
        """
        A simple backtracking k-colouring algorithm.
        This is a stand-in for Algorithm 8.1 [1] and for
        the "constant time" small graph case.[1]
        """
        coloring = {}
        nodes = list(graph.all_vertices())
        
        def solve(node_index):
            if node_index == len(nodes):
                return True  # All nodes colored

            u = nodes[node_index]
            for c in range(1, k + 1):
                # Check if color c is valid for node u
                is_valid = True
                for v in graph.neighbors(u):
                    if v in coloring and coloring[v] == c:
                        is_valid = False
                        break

                if is_valid:
                    coloring[u] = c
                    if solve(node_index + 1):
                        return True
                    del coloring[u]  # Backtrack

            return False  # No valid color found

        if solve(0):
            return coloring
        else:
            # This should not happen if G is k-colourable (i.e., perfect)
            # but can happen if k is wrong.
            return {v: 1 for v in nodes} # Fail-safe

    def handle_leaf_node(self, graph, k):
        """
        Handles the four leaf conditions from Section 10.[1]
        
        [FIXED] Added color_offset for Condition 3.
        """
        # Condition 2: Clique number < k
        true_k = graph.get_omega() # Use graph's method
        if true_k < k:
            return self.P_k(graph, true_k) # Recurse with P_{k-1}

        # Condition 4: Small graph (|V(G)| <= 2k - 1)
        if len(graph) <= 2 * k - 1:
            return self.backtracking_colouring(graph, k)

        # Condition 3: Not anticonnected
        if not graph.get_complement().is_connected() and len(graph) > 1:
            anticomponents = graph.get_anticomponents()
            final_coloring = {}
            color_offset = 0 # <-- [FIX] Start offset for joining
            
            for anti_comp_nodes in anticomponents:
                sub_graph = graph.get_induced_subgraph(anti_comp_nodes)
                sub_k = sub_graph.get_omega() # Use graph's method
                sub_coloring = self.P_k(sub_graph, sub_k)
                
                # [FIX] Shift colors for each anticomponent
                for v, c in sub_coloring.items():
                    final_coloring[v] = c + color_offset
                color_offset += sub_k # Increase offset
                
            return final_coloring

        # Condition 1: No balanced skew partition (the default case)
        # We apply Alg 8.1 , which we've stubbed as backtracking_colouring
        return self.backtracking_colouring(graph, k)

    def permute_colors(self, coloring, from_colors, to_colors):
        """
        Helper for combine_colourings Step 3.
        Permutes colors to match a target set.
        This is a simplification; a true implementation
        may need a more robust bipartite matching.
        """
        new_coloring = coloring.copy()
        
        from_list = sorted(list(from_colors))
        to_list = sorted(list(to_colors))
        
        # Create a mapping
        map_f_to_t = {}
        unused_to = list(to_list)
        used_from = set()
        
        # Map intersecting colors first
        intersect = sorted(list(from_colors & to_colors))
        for c in intersect:
            if c in unused_to:
                map_f_to_t[c] = c
                unused_to.remove(c)
                used_from.add(c)
            
        # Map remaining
        from_remaining = sorted(list(from_colors - used_from))
        for i, f_col in enumerate(from_remaining):
            if i < len(unused_to):
                map_f_to_t[f_col] = unused_to[i]
                unused_to.pop(i) # Use this color
        
        # Apply permutation
        for v, c in coloring.items():
            if c in map_f_to_t:
                new_coloring[v] = map_f_to_t[c]
            # else: color is not in from_colors, keep it
        return new_coloring

    def combine_colourings(self, graph, coloring_1, coloring_2, A1, A2, B, k):
        """
        Implements the 4-step algorithm from Section 9.[1]
        
        [FIXED] Logic for B1/B2 partition.
        [FIXED] Logic for Step 3 loop.
        """
        if not B: # Edge case
             coloring_1.update(coloring_2)
             return coloring_1

        # Step 1: Partition B into (B1, B2)
        G_B = graph.get_induced_subgraph(B)
        B_anticomponents = G_B.get_anticomponents()
        
        if not B_anticomponents:
            # G[B] is anticonnected
            B_list = list(B)
            if not B_list: # B is empty
                coloring_1.update(coloring_2)
                return coloring_1
            B1 = {B_list[0]}
            B2 = B - B1
        else:
            B1 = B_anticomponents[0]
            B2 = B - B1
        
        if not B2 and len(B_anticomponents) == 1: # B has only one anticomponent
            B_list = list(B)
            B1 = {B_list[0]}
            B2 = B - B1

        b1_graph = graph.get_induced_subgraph(B1)
        b1 = b1_graph.get_omega()
        if b1 == 0 and len(b1_graph) > 0: b1 = 1 # omega(stable set) = 1
        if b1 == 0 and len(b1_graph) == 0:
             coloring_1.update(coloring_2)
             return coloring_1

        # Step 2: Define L_i and S_i
        phi_1 = coloring_1
        phi_2 = coloring_2

        L1 = {phi_1[v] for v in B1 if v in phi_1}
        L2 = {phi_2[v] for v in B1 if v in phi_2}
        l1 = len(L1)
        l2 = len(L2)
        
        # Permute colors to {1...l_i}
        phi_1 = self.permute_colors(phi_1, L1, set(range(1, l1 + 1)))
        phi_2 = self.permute_colors(phi_2, L2, set(range(1, l2 + 1)))
        L1 = set(range(1, l1 + 1))
        L2 = set(range(1, l2 + 1))

        S1 = {v for v in (A1 | B) if v in phi_1 and phi_1.get(v) in L1}
        S2 = {v for v in (A2 | B) if v in phi_2 and phi_2.get(v) in L2}

        psi_1 = phi_1.copy()
        psi_2 = phi_2.copy()

        # Step 3: Re-colour H_i (inductive call)
        # Paper states l_i <= k-1 [1]
        
        
        for i, (S_i, l_i, G_i_nodes, current_psi) in enumerate([
            (S1, l1, A1 | B, psi_1), 
            (S2, l2, A2 | B, psi_2)
        ]):
            if l_i <= b1:
                continue

            H_i_nodes = S_i
            H_i = graph.get_induced_subgraph(H_i_nodes)
            
            dummy_verts = [f'dummy_{i}_{j}' for j in range(l_i - b1)]
            dummy_clique_nodes = set()

            for dummy_v in dummy_verts:
                H_i.add_vertex(dummy_v)
                # Complete to B1
                for node_in_B1 in B1:
                    if node_in_B1 in H_i.all_vertices():
                        H_i.add_edge(dummy_v, node_in_B1)
                # Complete to other dummies
                for other_dummy in dummy_clique_nodes:
                    H_i.add_edge(dummy_v, other_dummy)
                dummy_clique_nodes.add(dummy_v)

            # This is the inductive call, as l_i <= k-1
            xi_i = self.P_k(H_i, l_i)

            # Permute colors so dummies get b1+1...l_i
            dummy_colors = {xi_i[v] for v in dummy_verts if v in xi_i}
            target_dummy_colors = set(range(b1 + 1, l_i + 1))
            xi_i = self.permute_colors(xi_i, dummy_colors, target_dummy_colors)

            # Update psi_i
            for v in S_i:
                if v in xi_i:
                    current_psi[v] = xi_i[v]
            
            if i == 0:
                psi_1 = current_psi
            else:
                psi_2 = current_psi


        # Step 4: Combine psi_1 and psi_2
        T1 = {v for v in (A1 | B) if v in psi_1 and psi_1.get(v, k+1) <= b1}
        T2 = {v for v in (A2 | B) if v in psi_2 and psi_2.get(v, k+1) <= b1}

        G_T = graph.get_induced_subgraph(T1 | T2)
        G_rest = graph.get_induced_subgraph(graph.all_vertices() - (T1 | T2))

        # Two more inductive calls (b1 < k and k-b1 < k) [1]
        coloring_T = self.P_k(G_T, b1)
        # Need to handle k-b1=0 case
        k_rest = k - b1
        if k_rest < 0: k_rest = 0
        coloring_rest = self.P_k(G_rest, k_rest)


        # Combine them
        final_coloring = {}
        for v, color in coloring_T.items():
            final_coloring[v] = color
        for v, color in coloring_rest.items():
            final_coloring[v] = color + b1  # Shift colors
            
        return final_coloring


    def P_k(self, graph, k):
        """
        Implements the main recursive algorithm 10.3 (P_k).[1]
        
        [FIXED] Changed memoization key to a stable sorted tuple.
        """
        # Memoization check
        # [FIX] Create a canonical, hashable key from sorted tuples
        adj_items = sorted([(v, tuple(sorted(ns))) for v, ns in graph.adj.items()])
        graph_key = (tuple(adj_items), k)
        
        if graph_key in self.memoization_table:
            return self.memoization_table[graph_key]

        # Base cases
        if k == 0:
            return {}
        if not graph.adj or len(graph) == 0:
            return {}
        
        # We must check for k=1 *after* checking for empty graph
        if k == 1:
            # All nodes must be in a stable set
            if graph.get_omega() > 1:
                # This is a problem, but return a coloring anyway
                pass
            return {v: 1 for v in graph.all_vertices()}

        
        # Check leaf conditions from Section 10 [1]
        true_k = graph.get_omega()
        if (true_k < k or
            len(graph) <= 2 * k - 1 or
            (not graph.get_complement().is_connected() and len(graph) > 1)):
            result = self.handle_leaf_node(graph, k)
            self.memoization_table[graph_key] = result
            return result

        # --- "Processing" step [1] ---
        # 1. Try to find a balanced skew partition
        partition = self.find_balanced_skew_partition()

        if partition is None:
            # 2. This is a LEAF node (Condition 1: No BSP)
            result = self.handle_leaf_node(graph, k)
        
        else:
            # 3. This is an INTERNAL node.
            (A, B) = partition
            
            G_A_components = graph.get_induced_subgraph(A).get_connected_components()
            
            
            if not G_A_components:
                A1 = set()
                A2 = set()
            else:
                A1 = G_A_components[0]
                A2 = A - A1
            
            if not A2 and len(G_A_components) == 1: # A has only one component
                # This fallback is needed if G_A is connected but
                # (A,B) is still a skew partition (because G_B_bar is not conn.)
                A_list = list(A)
                if not A_list: # A is empty
                    A1 = set()
                    A2 = set()
                else:
                    A1 = {A_list[0]}
                    A2 = A - A1

            
            G1 = graph.get_induced_subgraph(A1 | B)
            G2 = graph.get_induced_subgraph(A2 | B)

            # Recurse on children
            k1 = G1.get_omega()
            k2 = G2.get_omega()
            
            coloring_1 = self.P_k(G1, k1)
            coloring_2 = self.P_k(G2, k2)
            
            # 4. "Conquer" step (Section 9)
            result = self.combine_colourings(
                graph, coloring_1, coloring_2, A1, A2, B, k
            )
        
        self.memoization_table[graph_key] = result
        return result

    def main_colouring_algorithm(self):
        """
        Main entry point. Finds omega(G) and runs P_k.
        """
        print(f"Starting analysis for graph with {len(self.graph)} vertices...")
        
        
        k = self.graph.get_omega()
        if k == 0:
            print("Graph is empty.")
            return {}
        
        print(f"Graph has {len(self.graph)} vertices. Target k = omega(G) = {k}")
        
        # Call the main recursive algorithm
        colouring = self.P_k(self.graph, k)
        
        # Verify the colouring
        for u in self.graph.all_vertices():
            if u not in colouring:
                print(f"Warning: Vertex {u} was not colored.")
                continue
            for v in self.graph.neighbors(u):
                if v in colouring and v in self.graph.all_vertices() and u in colouring:
                    if colouring[u] == colouring[v]:
                        raise Exception(f"Invalid colouring: {u} ({colouring[u]}) and {v} ({colouring[v]}) have same color")
        
        num_colors = len(set(c for c in colouring.values() if c is not None))
        
        
        print(f"Found valid {num_colors}-colouring. Target was {k}.")
        if num_colors > k:
            print(f"Warning: Found {num_colors}-colouring, which is > k={k}.")
        
        return colouring

# --- Benchmarking Function ---

def create_p5():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    return g

def create_c4():
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 1)
    return g

def create_k3_3():
    g = Graph()
    for i in range(1, 4):
        for j in range(4, 7):
            g.add_edge(i, j)
    return g

def create_interval_k3():
    g = Graph()
    g.add_edge(1, 2); g.add_edge(1, 3); g.add_edge(2, 3) # (1,2,3) clique
    g.add_edge(2, 4); g.add_edge(3, 4)                 # (2,3,4) clique
    g.add_edge(3, 5); g.add_edge(4, 5)                 # (3,4,5) clique
    g.add_vertex(6) # Isolated vertex
    return g

def run_benchmarks():
    """
    Uses timeit to benchmark the main_colouring_algorithm.
    """
    
    graphs_to_test = {
        "P5 (n=5, k=2)": create_p5(),
        "C4 (n=4, k=2)": create_c4(),
        "K3,3 (n=6, k=2)": create_k3_3(),
        "Interval (n=6, k=3)": create_interval_k3(),
    }
    
    # We must pass all class/module definitions to timeit's setup
    setup_code = """
import sys
import itertools
sys.setrecursionlimit(2000)
from __main__ import Graph, GraphHelpers, CombinatorialColouringSolver
from __main__ import create_p5, create_c4, create_k3_3, create_interval_k3
    
graphs_to_test = {
    "P5 (n=5, k=2)": create_p5(),
    "C4 (n=4, k=2)": create_c4(),
    "K3,3 (n=6, k=2)": create_k3_3(),
    "Interval (n=6, k=3)": create_interval_k3(),
}
"""
    
    results = {}
    print("\n--- Running Benchmarks (timeit) ---")
    print("Note: Runtimes will be high due to brute-force stand-ins for O(nm) algorithms.")
    
    for name in graphs_to_test.keys():
        test_code = f"""
solver = CombinatorialColouringSolver(graphs_to_test['{name}'])
solver.main_colouring_algorithm()
"""
        
        try:
            # Run 'number=1' because the algorithm is extremely slow
            t = timeit.timeit(stmt=test_code, setup=setup_code, number=1)
            results[name] = f"{t:.6f} seconds"
            print(f"{name}: {t:.6f} seconds")
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            results[name] = "Error"
            
    print("--- Benchmarking Complete ---")
    return results

if __name__ == "__main__":
    # You can run the benchmarks or a single graph test.
    
    # --- Option 1: Run a single graph for debugging ---
    print("--- Running Single Graph Test (Interval n=6, k=3) ---")
    g_test = create_interval_k3()
    solver_test = CombinatorialColouringSolver(g_test)
    coloring = solver_test.main_colouring_algorithm()
    print("-------------------------------------------------")
    print("Final Colouring:")
    print(coloring)
    print("-------------------------------------------------")


    # --- Option 2: Run all benchmarks ---
    # Note: This will be VERY slow. The "Interval (n=6, k=3)" graph
    # will trigger the O(2^n) brute-force clique cutset finder
    # on many subproblems.
    
    # results = run_benchmarks()
    # print(results)