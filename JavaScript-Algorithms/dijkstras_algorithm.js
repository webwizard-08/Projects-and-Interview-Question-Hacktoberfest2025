/**
 * Implementation of Dijkstra's Algorithm in JavaScript
 * 
 * Dijkstra's algorithm finds the shortest path between nodes in a graph,
 * which can represent road networks, computer networks, or any weighted directed graph.
 * 
 * Time Complexity: O((V + E) log V) where V is number of vertices and E is number of edges
 * Space Complexity: O(V)
 */

class PriorityQueue {
    constructor() {
        this.values = [];
    }

    enqueue(val, priority) {
        this.values.push({ val, priority });
        this.sort();
    }

    dequeue() {
        return this.values.shift();
    }

    sort() {
        this.values.sort((a, b) => a.priority - b.priority);
    }
}

class WeightedGraph {
    constructor() {
        this.adjacencyList = {};
    }

    /**
     * Add a vertex to the graph
     * @param {string} vertex - The vertex to add
     */
    addVertex(vertex) {
        if (!this.adjacencyList[vertex]) {
            this.adjacencyList[vertex] = [];
        }
    }

    /**
     * Add an edge between two vertices with a weight
     * @param {string} vertex1 - First vertex
     * @param {string} vertex2 - Second vertex
     * @param {number} weight - Weight of the edge
     */
    addEdge(vertex1, vertex2, weight) {
        this.adjacencyList[vertex1].push({ node: vertex2, weight });
        this.adjacencyList[vertex2].push({ node: vertex1, weight });
    }

    /**
     * Find shortest path between start and finish vertices
     * @param {string} start - Starting vertex
     * @param {string} finish - Ending vertex
     * @returns {Object} Object containing shortest distance and path
     */
    dijkstra(start, finish) {
        const nodes = new PriorityQueue();
        const distances = {};
        const previous = {};
        const path = []; // to return at end
        let smallest;

        // Build up initial state
        for (let vertex in this.adjacencyList) {
            if (vertex === start) {
                distances[vertex] = 0;
                nodes.enqueue(vertex, 0);
            } else {
                distances[vertex] = Infinity;
                nodes.enqueue(vertex, Infinity);
            }
            previous[vertex] = null;
        }

        // As long as there is something to visit
        while (nodes.values.length) {
            smallest = nodes.dequeue().val;
            if (smallest === finish) {
                // We are done
                // Build up path to return
                while (previous[smallest]) {
                    path.push(smallest);
                    smallest = previous[smallest];
                }
                break;
            }

            if (smallest || distances[smallest] !== Infinity) {
                for (let neighbor in this.adjacencyList[smallest]) {
                    // Find neighboring node
                    let nextNode = this.adjacencyList[smallest][neighbor];
                    // Calculate new distance to neighboring node
                    let candidate = distances[smallest] + nextNode.weight;
                    let nextNeighbor = nextNode.node;
                    if (candidate < distances[nextNeighbor]) {
                        // Updating new smallest distance to neighbor
                        distances[nextNeighbor] = candidate;
                        // Updating previous - How we got to neighbor
                        previous[nextNeighbor] = smallest;
                        // Enqueue in priority queue with new priority
                        nodes.enqueue(nextNeighbor, candidate);
                    }
                }
            }
        }
        return {
            distance: distances[finish],
            path: path.concat(start).reverse()
        };
    }
}

// Example usage and test cases
function runTests() {
    // Create a new graph
    const graph = new WeightedGraph();

    // Add vertices
    ['A', 'B', 'C', 'D', 'E', 'F'].forEach(vertex => {
        graph.addVertex(vertex);
    });

    // Add edges
    graph.addEdge("A", "B", 4);
    graph.addEdge("A", "C", 2);
    graph.addEdge("B", "E", 3);
    graph.addEdge("C", "D", 2);
    graph.addEdge("C", "F", 4);
    graph.addEdge("D", "E", 3);
    graph.addEdge("D", "F", 1);
    graph.addEdge("E", "F", 1);

    // Test cases
    const testCases = [
        { start: "A", end: "E" },
        { start: "A", end: "F" },
        { start: "B", end: "F" }
    ];

    console.log("Running Dijkstra's Algorithm Test Cases:");
    console.log("Graph Structure:", JSON.stringify(graph.adjacencyList, null, 2));

    testCases.forEach(({ start, end }, index) => {
        console.log(`\nTest Case ${index + 1}:`);
        console.log(`Finding shortest path from ${start} to ${end}`);
        
        const result = graph.dijkstra(start, end);
        
        console.log("Shortest Distance:", result.distance);
        console.log("Path:", result.path.join(" -> "));
    });
}

// Run the tests
runTests();

module.exports = {
    WeightedGraph,
    PriorityQueue
};
