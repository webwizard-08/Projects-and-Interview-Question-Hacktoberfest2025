from collections import deque

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph.

    Args:
        graph (dict): The graph represented as an adjacency list.
        start_node: The starting node for the search.

    Returns:
        list: A list of visited nodes in BFS order.
    """
    visited = []
    queue = deque([start_node])
    visited.append(start_node)

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return visited
