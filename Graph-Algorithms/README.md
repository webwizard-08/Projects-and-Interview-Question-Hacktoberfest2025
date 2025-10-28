# 🌐 Graph Algorithms - Important Interview Problems

This folder contains comprehensive solutions to common graph-based interview problems across multiple programming languages.

## 🎯 What is a Graph?

A Graph is a non-linear data structure consisting of vertices (nodes) and edges that connect these vertices. Graphs are used to represent relationships between entities.

## 📁 Problems Covered

### Basic Graph Operations
1. **Graph Representation** - Adjacency List, Adjacency Matrix
2. **Breadth-First Search (BFS)** - Level-order traversal
3. **Depth-First Search (DFS)** - Recursive and iterative
4. **Detect Cycle in Directed Graph** - DFS with colors
5. **Detect Cycle in Undirected Graph** - Union-Find, DFS

### Shortest Path Algorithms
6. **Dijkstra's Algorithm** - Single-source shortest path
7. **Bellman-Ford Algorithm** - Handles negative weights
8. **Floyd-Warshall Algorithm** - All-pairs shortest path
9. **A* Search Algorithm** - Heuristic-based search
10. **Bidirectional BFS** - Two-way search

### Minimum Spanning Tree
11. **Kruskal's Algorithm** - Union-Find approach
12. **Prim's Algorithm** - Greedy approach
13. **Borůvka's Algorithm** - Parallel approach

### Advanced Graph Problems
14. **Topological Sorting** - Kahn's algorithm, DFS
15. **Strongly Connected Components** - Tarjan's algorithm
16. **Articulation Points** - Tarjan's algorithm
17. **Bridges in Graph** - Tarjan's algorithm
18. **Network Flow** - Ford-Fulkerson algorithm
19. **Bipartite Graph Check** - BFS/DFS coloring
20. **Hamiltonian Path/Cycle** - Backtracking

### Real-World Applications
21. **Social Network Analysis** - Friend suggestions
22. **Web Crawling** - BFS for web pages
23. **GPS Navigation** - Shortest path routing
24. **Dependency Resolution** - Topological sort
25. **Game AI** - Pathfinding algorithms

## 🛠️ Implementation Languages

- **Java** - Object-oriented approach with generics
- **Python** - Clean and readable implementations
- **C++** - STL-based solutions
- **JavaScript** - Modern ES6+ implementations

## ⏱️ Time Complexities

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| BFS/DFS   | O(V + E)       | O(V)            |
| Dijkstra  | O((V + E) log V)| O(V)            |
| Bellman-Ford | O(VE)        | O(V)            |
| Floyd-Warshall | O(V³)     | O(V²)           |
| Kruskal   | O(E log E)     | O(V)            |
| Prim      | O((V + E) log V)| O(V)           |
| Topological Sort | O(V + E) | O(V)            |

## 🚀 How to Run

### Java
```bash
javac *.java
java GraphAlgorithms
```

### Python
```bash
python graph_algorithms.py
```

### C++
```bash
g++ -o graph_algorithms *.cpp
./graph_algorithms
```

### JavaScript
```bash
node graph_algorithms.js
```

## 📖 Learning Resources

- [GeeksforGeeks - Graph Data Structure](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/)
- [LeetCode Graph Problems](https://leetcode.com/tag/graph/)
- [HackerRank Graph Challenges](https://www.hackerrank.com/domains/data-structures/graphs)

## 🤝 Contributing

Feel free to add more problems, improve existing solutions, or add implementations in other languages!

---

**Happy Coding! 🎉**
