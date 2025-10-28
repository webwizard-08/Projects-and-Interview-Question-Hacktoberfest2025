/*
 * 🌐 Graph Algorithms - Java Implementation
 * Author: AI Assistant
 * 
 * Description:
 * This Java program contains essential graph algorithms frequently asked
 * in technical interviews at top companies like FAANG, Google, Microsoft, and Amazon.
 * 
 * Each algorithm includes:
 *  - Problem definition
 *  - Clean Java implementation
 *  - Example usage
 *  - Time and Space Complexity
 */

import java.util.*;

public class GraphAlgorithms {
    
    // Graph representation using adjacency list
    static class Graph {
        private int V;
        private List<List<Integer>> adj;
        
        Graph(int V) {
            this.V = V;
            adj = new ArrayList<>();
            for (int i = 0; i < V; i++) {
                adj.add(new ArrayList<>());
            }
        }
        
        void addEdge(int u, int v) {
            adj.get(u).add(v);
        }
        
        void addUndirectedEdge(int u, int v) {
            adj.get(u).add(v);
            adj.get(v).add(u);
        }
        
        List<Integer> getNeighbors(int v) {
            return adj.get(v);
        }
        
        int getVertices() { return V; }
    }
    
    // 1️⃣ Breadth-First Search (BFS)
    public static void BFS(Graph graph, int start) {
        boolean[] visited = new boolean[graph.getVertices()];
        Queue<Integer> queue = new LinkedList<>();
        
        visited[start] = true;
        queue.offer(start);
        
        System.out.print("BFS traversal: ");
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            System.out.print(vertex + " ");
            
            for (int neighbor : graph.getNeighbors(vertex)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
        System.out.println();
    }
    
    // 2️⃣ Depth-First Search (DFS) - Recursive
    public static void DFS(Graph graph, int start) {
        boolean[] visited = new boolean[graph.getVertices()];
        System.out.print("DFS traversal: ");
        DFSRecursive(graph, start, visited);
        System.out.println();
    }
    
    private static void DFSRecursive(Graph graph, int vertex, boolean[] visited) {
        visited[vertex] = true;
        System.out.print(vertex + " ");
        
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                DFSRecursive(graph, neighbor, visited);
            }
        }
    }
    
    // 3️⃣ DFS - Iterative
    public static void DFSIterative(Graph graph, int start) {
        boolean[] visited = new boolean[graph.getVertices()];
        Stack<Integer> stack = new Stack<>();
        
        stack.push(start);
        System.out.print("DFS (Iterative): ");
        
        while (!stack.isEmpty()) {
            int vertex = stack.pop();
            if (!visited[vertex]) {
                visited[vertex] = true;
                System.out.print(vertex + " ");
                
                for (int neighbor : graph.getNeighbors(vertex)) {
                    if (!visited[neighbor]) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        System.out.println();
    }
    
    // 4️⃣ Detect Cycle in Directed Graph (DFS with colors)
    public static boolean hasCycleDirected(Graph graph) {
        int[] color = new int[graph.getVertices()]; // 0: white, 1: gray, 2: black
        
        for (int i = 0; i < graph.getVertices(); i++) {
            if (color[i] == 0 && hasCycleDFS(graph, i, color)) {
                return true;
            }
        }
        return false;
    }
    
    private static boolean hasCycleDFS(Graph graph, int vertex, int[] color) {
        color[vertex] = 1; // gray
        
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (color[neighbor] == 1) return true; // back edge
            if (color[neighbor] == 0 && hasCycleDFS(graph, neighbor, color)) {
                return true;
            }
        }
        
        color[vertex] = 2; // black
        return false;
    }
    
    // 5️⃣ Detect Cycle in Undirected Graph
    public static boolean hasCycleUndirected(Graph graph) {
        boolean[] visited = new boolean[graph.getVertices()];
        
        for (int i = 0; i < graph.getVertices(); i++) {
            if (!visited[i] && hasCycleUndirectedDFS(graph, i, -1, visited)) {
                return true;
            }
        }
        return false;
    }
    
    private static boolean hasCycleUndirectedDFS(Graph graph, int vertex, int parent, boolean[] visited) {
        visited[vertex] = true;
        
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                if (hasCycleUndirectedDFS(graph, neighbor, vertex, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true; // back edge
            }
        }
        return false;
    }
    
    // 6️⃣ Topological Sorting (Kahn's Algorithm)
    public static List<Integer> topologicalSort(Graph graph) {
        int[] inDegree = new int[graph.getVertices()];
        
        // Calculate in-degrees
        for (int i = 0; i < graph.getVertices(); i++) {
            for (int neighbor : graph.getNeighbors(i)) {
                inDegree[neighbor]++;
            }
        }
        
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < graph.getVertices(); i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        
        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            result.add(vertex);
            
            for (int neighbor : graph.getNeighbors(vertex)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        
        return result.size() == graph.getVertices() ? result : new ArrayList<>();
    }
    
    // 7️⃣ Shortest Path - BFS (unweighted graph)
    public static int[] shortestPathBFS(Graph graph, int start) {
        int[] distance = new int[graph.getVertices()];
        Arrays.fill(distance, -1);
        
        Queue<Integer> queue = new LinkedList<>();
        distance[start] = 0;
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            
            for (int neighbor : graph.getNeighbors(vertex)) {
                if (distance[neighbor] == -1) {
                    distance[neighbor] = distance[vertex] + 1;
                    queue.offer(neighbor);
                }
            }
        }
        
        return distance;
    }
    
    // 8️⃣ Dijkstra's Algorithm (weighted graph)
    static class Edge {
        int to, weight;
        Edge(int to, int weight) {
            this.to = to;
            this.weight = weight;
        }
    }
    
    static class WeightedGraph {
        private int V;
        private List<List<Edge>> adj;
        
        WeightedGraph(int V) {
            this.V = V;
            adj = new ArrayList<>();
            for (int i = 0; i < V; i++) {
                adj.add(new ArrayList<>());
            }
        }
        
        void addEdge(int u, int v, int weight) {
            adj.get(u).add(new Edge(v, weight));
        }
        
        List<Edge> getNeighbors(int v) {
            return adj.get(v);
        }
        
        int getVertices() { return V; }
    }
    
    public static int[] dijkstra(WeightedGraph graph, int start) {
        int[] distance = new int[graph.getVertices()];
        Arrays.fill(distance, Integer.MAX_VALUE);
        distance[start] = 0;
        
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        pq.offer(new int[]{start, 0});
        
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int vertex = current[0];
            int dist = current[1];
            
            if (dist > distance[vertex]) continue;
            
            for (Edge edge : graph.getNeighbors(vertex)) {
                int newDist = distance[vertex] + edge.weight;
                if (newDist < distance[edge.to]) {
                    distance[edge.to] = newDist;
                    pq.offer(new int[]{edge.to, newDist});
                }
            }
        }
        
        return distance;
    }
    
    // 9️⃣ Bipartite Graph Check
    public static boolean isBipartite(Graph graph) {
        int[] color = new int[graph.getVertices()];
        Arrays.fill(color, -1);
        
        for (int i = 0; i < graph.getVertices(); i++) {
            if (color[i] == -1) {
                if (!isBipartiteBFS(graph, i, color)) {
                    return false;
                }
            }
        }
        return true;
    }
    
    private static boolean isBipartiteBFS(Graph graph, int start, int[] color) {
        Queue<Integer> queue = new LinkedList<>();
        color[start] = 0;
        queue.offer(start);
        
        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            
            for (int neighbor : graph.getNeighbors(vertex)) {
                if (color[neighbor] == -1) {
                    color[neighbor] = 1 - color[vertex];
                    queue.offer(neighbor);
                } else if (color[neighbor] == color[vertex]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // 🔟 Strongly Connected Components (Kosaraju's Algorithm)
    public static List<List<Integer>> stronglyConnectedComponents(Graph graph) {
        List<List<Integer>> result = new ArrayList<>();
        boolean[] visited = new boolean[graph.getVertices()];
        Stack<Integer> stack = new Stack<>();
        
        // Step 1: Fill stack with vertices in order of finishing times
        for (int i = 0; i < graph.getVertices(); i++) {
            if (!visited[i]) {
                fillOrder(graph, i, visited, stack);
            }
        }
        
        // Step 2: Create transpose graph
        Graph transpose = getTranspose(graph);
        
        // Step 3: Process vertices in reverse order
        Arrays.fill(visited, false);
        while (!stack.isEmpty()) {
            int vertex = stack.pop();
            if (!visited[vertex]) {
                List<Integer> component = new ArrayList<>();
                DFSUtil(transpose, vertex, visited, component);
                result.add(component);
            }
        }
        
        return result;
    }
    
    private static void fillOrder(Graph graph, int vertex, boolean[] visited, Stack<Integer> stack) {
        visited[vertex] = true;
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                fillOrder(graph, neighbor, visited, stack);
            }
        }
        stack.push(vertex);
    }
    
    private static Graph getTranspose(Graph graph) {
        Graph transpose = new Graph(graph.getVertices());
        for (int i = 0; i < graph.getVertices(); i++) {
            for (int neighbor : graph.getNeighbors(i)) {
                transpose.addEdge(neighbor, i);
            }
        }
        return transpose;
    }
    
    private static void DFSUtil(Graph graph, int vertex, boolean[] visited, List<Integer> component) {
        visited[vertex] = true;
        component.add(vertex);
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (!visited[neighbor]) {
                DFSUtil(graph, neighbor, visited, component);
            }
        }
    }
    
    // 🧭 Demonstration
    public static void main(String[] args) {
        // Create a sample graph
        Graph graph = new Graph(6);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 4);
        graph.addEdge(3, 5);
        graph.addEdge(4, 5);
        
        System.out.println("=== Graph Algorithms Demo ===");
        
        // BFS
        BFS(graph, 0);
        
        // DFS
        DFS(graph, 0);
        DFSIterative(graph, 0);
        
        // Cycle detection
        System.out.println("Has cycle (directed): " + hasCycleDirected(graph));
        System.out.println("Has cycle (undirected): " + hasCycleUndirected(graph));
        
        // Topological sort
        List<Integer> topoSort = topologicalSort(graph);
        System.out.println("Topological sort: " + topoSort);
        
        // Shortest path
        int[] distances = shortestPathBFS(graph, 0);
        System.out.println("Shortest distances from 0: " + Arrays.toString(distances));
        
        // Bipartite check
        System.out.println("Is bipartite: " + isBipartite(graph));
        
        // Strongly connected components
        List<List<Integer>> scc = stronglyConnectedComponents(graph);
        System.out.println("Strongly Connected Components: " + scc);
        
        // Weighted graph example
        WeightedGraph wGraph = new WeightedGraph(4);
        wGraph.addEdge(0, 1, 4);
        wGraph.addEdge(0, 2, 1);
        wGraph.addEdge(1, 3, 1);
        wGraph.addEdge(2, 1, 2);
        wGraph.addEdge(2, 3, 5);
        
        int[] dijkstraDistances = dijkstra(wGraph, 0);
        System.out.println("Dijkstra distances from 0: " + Arrays.toString(dijkstraDistances));
    }
}
