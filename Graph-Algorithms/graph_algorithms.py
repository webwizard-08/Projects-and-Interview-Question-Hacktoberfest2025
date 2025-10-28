"""
🌐 Graph Algorithms - Python Implementation
Author: AI Assistant

Description:
This Python program contains essential graph algorithms frequently asked
in technical interviews at top companies like FAANG, Google, Microsoft, and Amazon.

Each algorithm includes:
- Problem definition
- Clean Python implementation
- Example usage
- Time and Space Complexity
"""

from collections import defaultdict, deque
import heapq
from typing import List, Dict, Set, Tuple, Optional


class Graph:
    """Graph representation using adjacency list"""
    
    def __init__(self, vertices: int):
        self.V = vertices
        self.adj = defaultdict(list)
    
    def add_edge(self, u: int, v: int):
        """Add directed edge from u to v"""
        self.adj[u].append(v)
    
    def add_undirected_edge(self, u: int, v: int):
        """Add undirected edge between u and v"""
        self.adj[u].append(v)
        self.adj[v].append(u)
    
    def get_neighbors(self, v: int) -> List[int]:
        """Get neighbors of vertex v"""
        return self.adj[v]


class WeightedGraph:
    """Weighted graph representation"""
    
    def __init__(self, vertices: int):
        self.V = vertices
        self.adj = defaultdict(list)
    
    def add_edge(self, u: int, v: int, weight: int):
        """Add weighted edge from u to v"""
        self.adj[u].append((v, weight))


def bfs(graph: Graph, start: int) -> List[int]:
    """
    1️⃣ Breadth-First Search (BFS)
    Time: O(V + E), Space: O(V)
    """
    visited = [False] * graph.V
    queue = deque([start])
    visited[start] = True
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    
    return result


def dfs(graph: Graph, start: int) -> List[int]:
    """
    2️⃣ Depth-First Search (DFS) - Recursive
    Time: O(V + E), Space: O(V)
    """
    visited = [False] * graph.V
    result = []
    
    def dfs_util(vertex):
        visited[vertex] = True
        result.append(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                dfs_util(neighbor)
    
    dfs_util(start)
    return result


def dfs_iterative(graph: Graph, start: int) -> List[int]:
    """
    3️⃣ DFS - Iterative
    Time: O(V + E), Space: O(V)
    """
    visited = [False] * graph.V
    stack = [start]
    result = []
    
    while stack:
        vertex = stack.pop()
        if not visited[vertex]:
            visited[vertex] = True
            result.append(vertex)
            
            # Add neighbors in reverse order to maintain left-to-right traversal
            for neighbor in reversed(graph.get_neighbors(vertex)):
                if not visited[neighbor]:
                    stack.append(neighbor)
    
    return result


def has_cycle_directed(graph: Graph) -> bool:
    """
    4️⃣ Detect Cycle in Directed Graph (DFS with colors)
    Time: O(V + E), Space: O(V)
    """
    color = [0] * graph.V  # 0: white, 1: gray, 2: black
    
    def has_cycle_dfs(vertex):
        color[vertex] = 1  # gray
        
        for neighbor in graph.get_neighbors(vertex):
            if color[neighbor] == 1:  # back edge
                return True
            if color[neighbor] == 0 and has_cycle_dfs(neighbor):
                return True
        
        color[vertex] = 2  # black
        return False
    
    for i in range(graph.V):
        if color[i] == 0 and has_cycle_dfs(i):
            return True
    
    return False


def has_cycle_undirected(graph: Graph) -> bool:
    """
    5️⃣ Detect Cycle in Undirected Graph
    Time: O(V + E), Space: O(V)
    """
    visited = [False] * graph.V
    
    def has_cycle_dfs(vertex, parent):
        visited[vertex] = True
        
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                if has_cycle_dfs(neighbor, vertex):
                    return True
            elif neighbor != parent:
                return True  # back edge
        
        return False
    
    for i in range(graph.V):
        if not visited[i] and has_cycle_dfs(i, -1):
            return True
    
    return False


def topological_sort(graph: Graph) -> List[int]:
    """
    6️⃣ Topological Sorting (Kahn's Algorithm)
    Time: O(V + E), Space: O(V)
    """
    in_degree = [0] * graph.V
    
    # Calculate in-degrees
    for i in range(graph.V):
        for neighbor in graph.get_neighbors(i):
            in_degree[neighbor] += 1
    
    queue = deque()
    for i in range(graph.V):
        if in_degree[i] == 0:
            queue.append(i)
    
    result = []
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == graph.V else []


def shortest_path_bfs(graph: Graph, start: int) -> List[int]:
    """
    7️⃣ Shortest Path - BFS (unweighted graph)
    Time: O(V + E), Space: O(V)
    """
    distance = [-1] * graph.V
    queue = deque([start])
    distance[start] = 0
    
    while queue:
        vertex = queue.popleft()
        
        for neighbor in graph.get_neighbors(vertex):
            if distance[neighbor] == -1:
                distance[neighbor] = distance[vertex] + 1
                queue.append(neighbor)
    
    return distance


def dijkstra(graph: WeightedGraph, start: int) -> List[int]:
    """
    8️⃣ Dijkstra's Algorithm (weighted graph)
    Time: O((V + E) log V), Space: O(V)
    """
    distance = [float('inf')] * graph.V
    distance[start] = 0
    pq = [(0, start)]
    
    while pq:
        dist, vertex = heapq.heappop(pq)
        
        if dist > distance[vertex]:
            continue
        
        for neighbor, weight in graph.adj[vertex]:
            new_dist = distance[vertex] + weight
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distance


def is_bipartite(graph: Graph) -> bool:
    """
    9️⃣ Bipartite Graph Check
    Time: O(V + E), Space: O(V)
    """
    color = [-1] * graph.V
    
    def is_bipartite_bfs(start):
        queue = deque([start])
        color[start] = 0
        
        while queue:
            vertex = queue.popleft()
            
            for neighbor in graph.get_neighbors(vertex):
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    return False
        
        return True
    
    for i in range(graph.V):
        if color[i] == -1:
            if not is_bipartite_bfs(i):
                return False
    
    return True


def strongly_connected_components(graph: Graph) -> List[List[int]]:
    """
    🔟 Strongly Connected Components (Kosaraju's Algorithm)
    Time: O(V + E), Space: O(V)
    """
    def fill_order(vertex, visited, stack):
        visited[vertex] = True
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                fill_order(neighbor, visited, stack)
        stack.append(vertex)
    
    def get_transpose():
        transpose = Graph(graph.V)
        for i in range(graph.V):
            for neighbor in graph.get_neighbors(i):
                transpose.add_edge(neighbor, i)
        return transpose
    
    def dfs_util(vertex, visited, component):
        visited[vertex] = True
        component.append(vertex)
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                dfs_util(neighbor, visited, component)
    
    # Step 1: Fill stack with vertices in order of finishing times
    visited = [False] * graph.V
    stack = []
    for i in range(graph.V):
        if not visited[i]:
            fill_order(i, visited, stack)
    
    # Step 2: Create transpose graph
    transpose = get_transpose()
    
    # Step 3: Process vertices in reverse order
    visited = [False] * graph.V
    result = []
    while stack:
        vertex = stack.pop()
        if not visited[vertex]:
            component = []
            dfs_util(vertex, visited, component)
            result.append(component)
    
    return result


def kruskal_mst(graph: WeightedGraph) -> List[Tuple[int, int, int]]:
    """
    1️⃣1️⃣ Kruskal's Algorithm for MST
    Time: O(E log E), Space: O(V)
    """
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True
    
    # Collect all edges
    edges = []
    for u in range(graph.V):
        for v, weight in graph.adj[u]:
            if u < v:  # Avoid duplicate edges
                edges.append((weight, u, v))
    
    edges.sort()  # Sort by weight
    uf = UnionFind(graph.V)
    mst = []
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            if len(mst) == graph.V - 1:
                break
    
    return mst


def prim_mst(graph: WeightedGraph) -> List[Tuple[int, int, int]]:
    """
    1️⃣2️⃣ Prim's Algorithm for MST
    Time: O((V + E) log V), Space: O(V)
    """
    mst = []
    visited = [False] * graph.V
    pq = [(0, 0, -1)]  # (weight, vertex, parent)
    
    while pq and len(mst) < graph.V - 1:
        weight, vertex, parent = heapq.heappop(pq)
        
        if visited[vertex]:
            continue
        
        visited[vertex] = True
        if parent != -1:
            mst.append((parent, vertex, weight))
        
        for neighbor, edge_weight in graph.adj[vertex]:
            if not visited[neighbor]:
                heapq.heappush(pq, (edge_weight, neighbor, vertex))
    
    return mst


def articulation_points(graph: Graph) -> Set[int]:
    """
    1️⃣3️⃣ Find Articulation Points (Tarjan's Algorithm)
    Time: O(V + E), Space: O(V)
    """
    ap = set()
    visited = [False] * graph.V
    disc = [0] * graph.V
    low = [0] * graph.V
    time = [0]
    parent = [-1] * graph.V
    
    def dfs(vertex):
        visited[vertex] = True
        disc[vertex] = low[vertex] = time[0]
        time[0] += 1
        children = 0
        
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                children += 1
                parent[neighbor] = vertex
                dfs(neighbor)
                low[vertex] = min(low[vertex], low[neighbor])
                
                # Check if vertex is articulation point
                if parent[vertex] == -1 and children > 1:
                    ap.add(vertex)
                if parent[vertex] != -1 and low[neighbor] >= disc[vertex]:
                    ap.add(vertex)
            elif neighbor != parent[vertex]:
                low[vertex] = min(low[vertex], disc[neighbor])
    
    for i in range(graph.V):
        if not visited[i]:
            dfs(i)
    
    return ap


def bridges(graph: Graph) -> List[Tuple[int, int]]:
    """
    1️⃣4️⃣ Find Bridges (Tarjan's Algorithm)
    Time: O(V + E), Space: O(V)
    """
    bridges_list = []
    visited = [False] * graph.V
    disc = [0] * graph.V
    low = [0] * graph.V
    time = [0]
    parent = [-1] * graph.V
    
    def dfs(vertex):
        visited[vertex] = True
        disc[vertex] = low[vertex] = time[0]
        time[0] += 1
        
        for neighbor in graph.get_neighbors(vertex):
            if not visited[neighbor]:
                parent[neighbor] = vertex
                dfs(neighbor)
                low[vertex] = min(low[vertex], low[neighbor])
                
                # Check if edge is bridge
                if low[neighbor] > disc[vertex]:
                    bridges_list.append((vertex, neighbor))
            elif neighbor != parent[vertex]:
                low[vertex] = min(low[vertex], disc[neighbor])
    
    for i in range(graph.V):
        if not visited[i]:
            dfs(i)
    
    return bridges_list


def hamiltonian_path(graph: Graph) -> Optional[List[int]]:
    """
    1️⃣5️⃣ Hamiltonian Path (Backtracking)
    Time: O(V!), Space: O(V)
    """
    def is_valid_path(path):
        for i in range(len(path) - 1):
            if path[i + 1] not in graph.get_neighbors(path[i]):
                return False
        return True
    
    def hamiltonian_util(path, visited):
        if len(path) == graph.V:
            return path[:]
        
        last_vertex = path[-1]
        for neighbor in graph.get_neighbors(last_vertex):
            if not visited[neighbor]:
                visited[neighbor] = True
                path.append(neighbor)
                
                result = hamiltonian_util(path, visited)
                if result:
                    return result
                
                path.pop()
                visited[neighbor] = False
        
        return None
    
    for start in range(graph.V):
        visited = [False] * graph.V
        visited[start] = True
        result = hamiltonian_util([start], visited)
        if result:
            return result
    
    return None


# 🧭 Demonstration
def main():
    print("=== Graph Algorithms Demo ===")
    
    # Create a sample graph
    graph = Graph(6)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)
    
    # BFS
    print(f"BFS traversal: {bfs(graph, 0)}")
    
    # DFS
    print(f"DFS traversal: {dfs(graph, 0)}")
    print(f"DFS (Iterative): {dfs_iterative(graph, 0)}")
    
    # Cycle detection
    print(f"Has cycle (directed): {has_cycle_directed(graph)}")
    print(f"Has cycle (undirected): {has_cycle_undirected(graph)}")
    
    # Topological sort
    print(f"Topological sort: {topological_sort(graph)}")
    
    # Shortest path
    print(f"Shortest distances from 0: {shortest_path_bfs(graph, 0)}")
    
    # Bipartite check
    print(f"Is bipartite: {is_bipartite(graph)}")
    
    # Strongly connected components
    print(f"Strongly Connected Components: {strongly_connected_components(graph)}")
    
    # Weighted graph example
    w_graph = WeightedGraph(4)
    w_graph.add_edge(0, 1, 4)
    w_graph.add_edge(0, 2, 1)
    w_graph.add_edge(1, 3, 1)
    w_graph.add_edge(2, 1, 2)
    w_graph.add_edge(2, 3, 5)
    
    print(f"Dijkstra distances from 0: {dijkstra(w_graph, 0)}")
    
    # MST algorithms
    print(f"Kruskal MST: {kruskal_mst(w_graph)}")
    print(f"Prim MST: {prim_mst(w_graph)}")
    
    # Articulation points and bridges
    undirected_graph = Graph(5)
    undirected_graph.add_undirected_edge(0, 1)
    undirected_graph.add_undirected_edge(1, 2)
    undirected_graph.add_undirected_edge(2, 3)
    undirected_graph.add_undirected_edge(3, 4)
    undirected_graph.add_undirected_edge(1, 3)
    
    print(f"Articulation points: {articulation_points(undirected_graph)}")
    print(f"Bridges: {bridges(undirected_graph)}")
    
    # Hamiltonian path
    hamiltonian_graph = Graph(4)
    hamiltonian_graph.add_undirected_edge(0, 1)
    hamiltonian_graph.add_undirected_edge(1, 2)
    hamiltonian_graph.add_undirected_edge(2, 3)
    hamiltonian_graph.add_undirected_edge(3, 0)
    hamiltonian_graph.add_undirected_edge(0, 2)
    
    print(f"Hamiltonian path: {hamiltonian_path(hamiltonian_graph)}")


if __name__ == "__main__":
    main()
