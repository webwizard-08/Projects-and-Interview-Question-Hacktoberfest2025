import java.util.*;

/**
 * Implements a directed Graph using an adjacency list representation.
 * Includes methods for adding edges and performing Breadth-First Search (BFS)
 * and Depth-First Search (DFS) traversals.
 */
public class Graph {

  private final int V; // Number of vertices
  private final List<List<Integer>> adj; // Adjacency List

  /**
   * Constructor for the Graph.
   * 
   * @param v Number of vertices in the graph.
   */
  public Graph(int v) {
    V = v;
    adj = new ArrayList<>(v);
    for (int i = 0; i < v; ++i) {
      adj.add(new ArrayList<>());
    }
  }

  /**
   * Adds a directed edge from vertex v to vertex w.
   * 
   * @param v The source vertex.
   * @param w The destination vertex.
   */
  public void addEdge(int v, int w) {
    adj.get(v).add(w);
  }

  /**
   * Performs a Breadth-First Search (BFS) traversal starting from a given vertex.
   * 
   * @param s The starting vertex for the traversal.
   */
  public void BFS(int s) {
    System.out.print("BFS starting from vertex " + s + ": ");
    boolean[] visited = new boolean[V];
    Queue<Integer> queue = new LinkedList<>();

    visited[s] = true;
    queue.add(s);

    while (!queue.isEmpty()) {
      s = queue.poll();
      System.out.print(s + " ");

      for (int n : adj.get(s)) {
        if (!visited[n]) {
          visited[n] = true;
          queue.add(n);
        }
      }
    }
    System.out.println();
  }

  // Helper function for DFS traversal
  private void DFSUtil(int v, boolean[] visited) {
    visited[v] = true;
    System.out.print(v + " ");

    for (int n : adj.get(v)) {
      if (!visited[n]) {
        DFSUtil(n, visited);
      }
    }
  }

  /**
   * Performs a Depth-First Search (DFS) traversal starting from a given vertex.
   * 
   * @param v The starting vertex for the traversal.
   */
  public void DFS(int v) {
    System.out.print("DFS starting from vertex " + v + ": ");
    boolean[] visited = new boolean[V];
    DFSUtil(v, visited);
    System.out.println();
  }

  // Main method to test the graph implementation
  public static void main(String[] args) {
    Graph g = new Graph(4);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    g.BFS(2); // Expected: 2 0 3 1
    g.DFS(2); // Expected: 2 0 1 3
  }
}