package bellmanford;

import java.util.Scanner;
import java.util.Arrays;
public class ShortestPath {
    static final int INF = 999;
    public static void pathVector(int v, int[][] graph){
        int [][] dist = new int[v][v];
        String [][] path = new String [v][v];
        for(int i=0; i<v; i++) {
            for(int j=0; j<v; j++){
                dist[i][j]= graph[i][j];
                path[i][j] = (graph[i][j]==INF)? "inf": i+"->"+j;
                if(i==j) path[i][j]= String.valueOf(i);
            }
        }

        for(int k=0; k<v; k++){
            for(int i=0; i<v; i++){
                for(int j=0; j<v; j++){
                    if(dist[i][k]+dist[k][j]<dist[i][j]){
                        dist[i][j] =  dist[i][k]+dist[k][j];
                        path[i][j] = path[i][k] + path[k][j].substring(path[k][j].indexOf("->"));
                    }
                }
            }
        }
        System.out.println("Path vector routing table");
        for(int i=0; i<v; i++){
            for(int j=0; j<v; j++){
                System.out.println("From "+i+" to "+j+"="+ path[i][j]);
            }
        }
    }
    public static void bellmanFord(int v, int e, int[][] edges, int src ){
        int[] dist = new int[v];
        Arrays.fill(dist, INF);
        dist[src]=0;

        for(int i=1; i<v; i++){
            for(int j=0; j<e; j++){
                int u = edges[j][0];
                int v1 = edges[j][1];
                int w = edges[j][2];

                if(dist[u]+w < dist[v1]){
                    dist[v1] = dist[u]+w;
                }

            }
        }

        boolean negativeEdgeCycle = false;
        for(int j=0; j<e; j++){
            int u = edges[j][0];
            int v1 = edges[j][1];
            int w = edges[j][2];

            if(dist[u]+w < dist[v1]){
                negativeEdgeCycle = true;
                break;
            }
        }

        if(negativeEdgeCycle) {
            System.out.println("The graph contains negative edge cycle");
        }else{
            System.out.println("Bellman ford distance from "+src);
            for(int i=0; i<v; i++){
                System.out.println("To "+i+"="+dist[i]);
            }
        }

    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the number of vertices: ");
        int v = sc.nextInt();
        int[][] graph = new int[v][v];
        System.out.println("Enter the cost adjacency matrix (999 for no direct edge)");
        for(int i=0; i<v; i++){
            for(int j=0; j<v; j++){
                graph[i][j] = sc.nextInt();
            }
        }

        pathVector(v, graph);
        System.out.print("Enter the number of edges: ");
        int e = sc.nextInt();
        int[][] edges = new int[e][3];
        System.out.println("Enter the weight from u to v format: u v w");
        for(int i=0; i<e; i++){
            edges[i][0] = sc.nextInt();
            edges[i][1] = sc.nextInt();
            edges[i][2] = sc.nextInt();
        }
        System.out.println("Enter the source vertex: ");
        bellmanFord(v, e, edges, sc.nextInt());
    }

}
