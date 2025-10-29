from collections import defaultdict, deque
from typing import Dict, List, Set

class Graph:
    """
    Graph implementation with DFS and BFS traversal methods.
    Uses adjacency list representation.
    """
    
    def __init__(self):
        """Initialize an empty graph using defaultdict for adjacency list"""
        self.graph = defaultdict(list)
        
    def add_edge(self, u: int, v: int):
        """
        Add an edge to the graph
        
        Args:
            u (int): Source vertex
            v (int): Destination vertex
        """
        self.graph[u].append(v)
        
    def bfs(self, start: int) -> List[int]:
        """
        Perform Breadth First Search traversal from start vertex
        
        Args:
            start (int): Starting vertex
            
        Returns:
            List[int]: List of vertices in BFS order
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # Add unvisited neighbors to queue
                queue.extend(v for v in self.graph[vertex] if v not in visited)
                
        return result
        
    def dfs(self, start: int) -> List[int]:
        """
        Perform Depth First Search traversal from start vertex
        
        Args:
            start (int): Starting vertex
            
        Returns:
            List[int]: List of vertices in DFS order
            
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = set()
        result = []
        
        def dfs_recursive(vertex: int):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_recursive(neighbor)
                    
        dfs_recursive(start)
        return result

# Test cases
if __name__ == "__main__":
    # Test case 1: Simple graph
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)
    
    # Test BFS
    assert g.bfs(2) == [2, 0, 3, 1]
    
    # Test DFS
    assert g.dfs(2) == [2, 0, 1, 3]
    
    # Test case 2: Linear graph
    g2 = Graph()
    g2.add_edge(0, 1)
    g2.add_edge(1, 2)
    g2.add_edge(2, 3)
    
    assert g2.bfs(0) == [0, 1, 2, 3]
    assert g2.dfs(0) == [0, 1, 2, 3]
    
    # Test case 3: Disconnected vertex
    g3 = Graph()
    g3.add_edge(0, 1)
    g3.add_edge(2, 2)  # Self loop
    
    assert g3.bfs(0) == [0, 1]
    assert g3.dfs(0) == [0, 1]
    
    print("All graph traversal tests passed!")