import sys
from collections import deque

# Increase recursion depth for deep recursions in decomposition if necessary
sys.setrecursionlimit(3000)

class ManualGraph:
    """
    A manual implementation of a Graph data structure using Adjacency Matrices.
    Strictly avoids NetworkX/iGraph logic to simulate pure combinatorial overhead.
    """
    def __init__(self, num_vertices=0, matrix=None):
        if matrix is not None:
            self.n = len(matrix)
            # Deep copy to ensure safety
            self.adj = [row[:] for row in matrix] 
            self.vertices = list(range(self.n))
        else:
            self.n = num_vertices
            self.adj = [ * num_vertices for _ in range(num_vertices)]
            self.vertices = list(range(num_vertices))

    def neighbors(self, u):
        """Return list of neighbors for vertex u."""
        return [v for v, is_connected in enumerate(self.adj[u]) if is_connected]

    def has_edge(self, u, v):
        return self.adj[u][v] == 1

    def subgraph(self, subset_indices):
        """
        Returns the induced subgraph for a set of vertex indices.
        Complexity: O(k^2) where k is len(subset_indices).
        """
        subset_indices = sorted(list(subset_indices))
        new_n = len(subset_indices)
        
        # Create new empty matrix
        new_matrix = [ * new_n for _ in range(new_n)]
        
        for i in range(new_n):
            for j in range(new_n):
                u_old = subset_indices[i]
                v_old = subset_indices[j]
                if self.adj[u_old][v_old] == 1:
                    new_matrix[i][j] = 1
                    
        return ManualGraph(matrix=new_matrix)

    def complement(self):
        """
        Returns the complement graph (edges <-> non-edges).
        Complexity: O(N^2).
        """
        comp_matrix = [ * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i!= j and self.adj[i][j] == 0:
                    comp_matrix[i][j] = 1
        return ManualGraph(matrix=comp_matrix)

    def is_connected(self):
        """
        Manual BFS to check connectivity.
        Complexity: O(V + E).
        """
        if self.n == 0: return True
        
        visited = [False] * self.n
        start_node = 0
        visited[start_node] = True
        queue = deque([start_node])
        count = 1
        
        while queue:
            u = queue.popleft()
            for v, is_neighbor in enumerate(self.adj[u]):
                if is_neighbor and not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == self.n

# --- Algorithm 1: Star Cutset Detection (O(N^3)) ---

def find_star_cutset(graph):
    """
    Detects if a Star Cutset exists.
    A star cutset is a set C where G \ C is disconnected, 
    and C contains a vertex 'center' adjacent to all others in C.
    """
    nodes = graph.vertices
    
    # Iterate over every vertex assuming it is the center of the star
    for v in nodes:
        # Candidate cutset C = {v} U Neighbors(v) (Closed Neighborhood)
        star_neighbors = graph.neighbors(v)
        cutset = set(star_neighbors)
        cutset.add(v)
        
        # Remainder vertices R = V \ C
        remainder = [u for u in nodes if u not in cutset]
        
        # If cutset is everything, it can't disconnect the graph (trivial case)
        if not remainder:
            continue
            
        # Check if R induces a disconnected subgraph
        sub_g = graph.subgraph(remainder)
        if not sub_g.is_connected():
            return True, v # Found one
            
    return False, None

# --- Algorithm 2: Tight Skew Partition Detection (O(N^6)) ---

def find_square_based_skew_partition(graph):
    """
    Implements 'Square-Based' Skew Partition detection (Chudnovsky et al., 2017).
    
    Logic:
    1. Iterate all induced cycles of length 4 (a-b-c-d-a).
    2. Construct B = (N(a) n N(c)) U (N(b) n N(d)).
    3. Construct A = V \ B.
    4. Check if G[A] is disconnected AND Complement(G) is disconnected.
    """
    n = graph.n
    nodes = graph.vertices
    
    # Optimization: Iterating 4 loops is O(N^4).
    # We look for a-b-c-d such that edges are (a,b), (b,c), (c,d), (d,a)
    # and NO chords (a,c) or (b,d).
    
    for a in range(n):
        for b in graph.neighbors(a):
            for c in graph.neighbors(b):
                # Check induced condition early: c cannot be a or neighbor of a
                if c == a or graph.has_edge(a, c): 
                    continue
                
                for d in graph.neighbors(c):
                    if d == b: continue # Backtracking to b
                    if not graph.has_edge(d, a): continue # Must close cycle
                    if graph.has_edge(d, b): continue # Check chord b-d
                    
                    # If we are here, {a,b,c,d} induces a C4.
                    
                    # --- Construct B ---
                    # B = (N(a) n N(c)) U (N(b) n N(d))
                    Na = set(graph.neighbors(a))
                    Nb = set(graph.neighbors(b))
                    Nc = set(graph.neighbors(c))
                    Nd = set(graph.neighbors(d))
                    
                    group1 = Na.intersection(Nc)
                    group2 = Nb.intersection(Nd)
                    
                    B_set = group1.union(group2)
                    A_set = set(nodes) - B_set
                    
                    # --- Verify Skew Partition Properties (The Bottleneck) ---
                    
                    # 1. G[A] must be disconnected
                    if len(A_set) < 2: continue 
                    sub_A = graph.subgraph(A_set)
                    if sub_A.is_connected():
                        # If A is connected, this is not a skew partition
                        continue
                        
                    # 2. Complement(G) must be disconnected
                    if len(B_set) < 2: continue
                    sub_B_in_G = graph.subgraph(B_set)
                    sub_B_comp = sub_B_in_G.complement()
                    
                    if sub_B_comp.is_connected():
                        # If B is connected in complement, not a skew partition
                        continue
                        
                    # If we pass both, we found a valid skew partition
                    return True, (list(A_set), list(B_set))
                    
    return False, None