/*
 * Dijkstra's Shortest Path Algorithm Implementation
 * 
 * Description:
 * Dijkstra's algorithm is a graph search algorithm that finds the shortest path
 * between nodes in a graph with non-negative edge weights. It is widely used in
 * network routing protocols, GPS navigation, and many other applications.
 * 
 * Time Complexity: O((V + E) log V) with priority queue
 * Space Complexity: O(V)
 * 
 * Author: Contributing to Hacktoberfest 2025
 * Date: October 2025
 */

#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>
using namespace std;

// Structure to represent an edge in the graph
struct Edge {
    int destination;
    int weight;
    
    Edge(int dest, int w) : destination(dest), weight(w) {}
};

// Comparator for priority queue (min-heap based on distance)
struct Compare {
    bool operator()(const pair<int, int>& a, const pair<int, int>& b) {
        return a.second > b.second;
    }
};

class Graph {
private:
    int vertices;
    vector<vector<Edge>> adjacencyList;
    
public:
    // Constructor
    Graph(int v) : vertices(v) {
        adjacencyList.resize(v);
    }
    
    // Add an edge to the graph
    void addEdge(int source, int destination, int weight) {
        adjacencyList[source].push_back(Edge(destination, weight));
        // For undirected graph, uncomment the line below:
        // adjacencyList[destination].push_back(Edge(source, weight));
    }
    
    /**
     * Dijkstra's Algorithm Implementation
     * 
     * @param source: Starting vertex
     * @return: Vector containing shortest distances from source to all vertices
     */
    vector<int> dijkstra(int source) {
        // Initialize distances to infinity
        vector<int> distance(vertices, INT_MAX);
        
        // Priority queue to store {distance, vertex}
        priority_queue<pair<int, int>, vector<pair<int, int>>, Compare> pq;
        
        // Distance from source to itself is 0
        distance[source] = 0;
        pq.push({source, 0});
        
        // Process vertices
        while (!pq.empty()) {
            int currentVertex = pq.top().first;
            int currentDistance = pq.top().second;
            pq.pop();
            
            // Skip if we've already found a shorter path
            if (currentDistance > distance[currentVertex]) {
                continue;
            }
            
            // Check all adjacent vertices
            for (const Edge& edge : adjacencyList[currentVertex]) {
                int neighbor = edge.destination;
                int weight = edge.weight;
                
                // Calculate new distance
                int newDistance = distance[currentVertex] + weight;
                
                // Update if shorter path found
                if (newDistance < distance[neighbor]) {
                    distance[neighbor] = newDistance;
                    pq.push({neighbor, newDistance});
                }
            }
        }
        
        return distance;
    }
    
    /**
     * Dijkstra's Algorithm with Path Reconstruction
     * 
     * @param source: Starting vertex
     * @param destination: Ending vertex
     * @return: Pair containing {shortest distance, path}
     */
    pair<int, vector<int>> dijkstraWithPath(int source, int destination) {
        vector<int> distance(vertices, INT_MAX);
        vector<int> parent(vertices, -1);
        priority_queue<pair<int, int>, vector<pair<int, int>>, Compare> pq;
        
        distance[source] = 0;
        pq.push({source, 0});
        
        while (!pq.empty()) {
            int currentVertex = pq.top().first;
            int currentDistance = pq.top().second;
            pq.pop();
            
            if (currentDistance > distance[currentVertex]) {
                continue;
            }
            
            for (const Edge& edge : adjacencyList[currentVertex]) {
                int neighbor = edge.destination;
                int weight = edge.weight;
                int newDistance = distance[currentVertex] + weight;
                
                if (newDistance < distance[neighbor]) {
                    distance[neighbor] = newDistance;
                    parent[neighbor] = currentVertex;
                    pq.push({neighbor, newDistance});
                }
            }
        }
        
        // Reconstruct path
        vector<int> path;
        if (distance[destination] != INT_MAX) {
            int current = destination;
            while (current != -1) {
                path.push_back(current);
                current = parent[current];
            }
            reverse(path.begin(), path.end());
        }
        
        return {distance[destination], path};
    }
    
    // Print the shortest distances from source
    void printDistances(const vector<int>& distances, int source) {
        cout << "\nShortest distances from vertex " << source << ":\n";
        cout << "Vertex\tDistance from Source\n";
        for (int i = 0; i < vertices; i++) {
            cout << i << "\t";
            if (distances[i] == INT_MAX) {
                cout << "INF\n";
            } else {
                cout << distances[i] << "\n";
            }
        }
    }
};

// Test case 1: Simple graph
void testCase1() {
    cout << "\n========== Test Case 1: Simple Graph ==========\n";
    Graph g(5);
    
    // Building a simple graph
    g.addEdge(0, 1, 4);
    g.addEdge(0, 2, 1);
    g.addEdge(2, 1, 2);
    g.addEdge(1, 3, 1);
    g.addEdge(2, 3, 5);
    g.addEdge(3, 4, 3);
    
    int source = 0;
    vector<int> distances = g.dijkstra(source);
    g.printDistances(distances, source);
    
    // Test path reconstruction
    auto result = g.dijkstraWithPath(0, 4);
    cout << "\nShortest path from 0 to 4:\n";
    cout << "Distance: " << result.first << "\n";
    cout << "Path: ";
    for (int vertex : result.second) {
        cout << vertex << " ";
    }
    cout << "\n";
}

// Test case 2: Disconnected graph
void testCase2() {
    cout << "\n========== Test Case 2: Disconnected Graph ==========\n";
    Graph g(4);
    
    g.addEdge(0, 1, 5);
    g.addEdge(1, 2, 3);
    // Vertex 3 is disconnected
    
    int source = 0;
    vector<int> distances = g.dijkstra(source);
    g.printDistances(distances, source);
}

// Test case 3: Complex graph with multiple paths
void testCase3() {
    cout << "\n========== Test Case 3: Complex Graph ==========\n";
    Graph g(6);
    
    g.addEdge(0, 1, 2);
    g.addEdge(0, 2, 4);
    g.addEdge(1, 2, 1);
    g.addEdge(1, 3, 7);
    g.addEdge(2, 4, 3);
    g.addEdge(3, 4, 2);
    g.addEdge(3, 5, 1);
    g.addEdge(4, 5, 5);
    
    int source = 0;
    vector<int> distances = g.dijkstra(source);
    g.printDistances(distances, source);
    
    // Test multiple path reconstructions
    cout << "\nPath Analysis:\n";
    for (int dest = 1; dest < 6; dest++) {
        auto result = g.dijkstraWithPath(0, dest);
        cout << "\nVertex 0 to " << dest << ":\n";
        cout << "  Distance: " << result.first << "\n";
        cout << "  Path: ";
        for (int vertex : result.second) {
            cout << vertex << " ";
        }
        cout << "\n";
    }
}

int main() {
    cout << "\n============================================\n";
    cout << "  Dijkstra's Shortest Path Algorithm Demo  \n";
    cout << "  Hacktoberfest 2025 Contribution         \n";
    cout << "============================================\n";
    
    // Run test cases
    testCase1();
    testCase2();
    testCase3();
    
    cout << "\n\n============================================\n";
    cout << "  All test cases completed successfully!   \n";
    cout << "============================================\n\n";
    
    return 0;
}

/*
 * Interview Questions and Discussion Points:
 * 
 * 1. Why can't Dijkstra's algorithm handle negative edge weights?
 *    - Because it makes a greedy choice and doesn't revisit vertices.
 *    - Use Bellman-Ford algorithm for graphs with negative weights.
 * 
 * 2. What are the differences between Dijkstra's and BFS?
 *    - BFS works for unweighted graphs (or graphs with equal weights)
 *    - Dijkstra's works for weighted graphs with non-negative weights
 * 
 * 3. Real-world applications:
 *    - GPS Navigation systems
 *    - Network routing protocols (OSPF)
 *    - Google Maps route planning
 *    - Social networking (shortest connection path)
 * 
 * 4. Optimization techniques:
 *    - Use Fibonacci heap for better time complexity: O(E + V log V)
 *    - Bidirectional Dijkstra for faster path finding
 *    - A* algorithm with heuristics for goal-directed search
 * 
 * 5. Common variations:
 *    - Single-source shortest path (implemented here)
 *    - All-pairs shortest path (use Floyd-Warshall instead)
 *    - K-shortest paths problem
 */
