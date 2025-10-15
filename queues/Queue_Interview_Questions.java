/*
 * 🧠 Top 20 Queue Interview Questions – Java Implementation
 * Author: Sai Surya
 *
 * Description:
 * This Java program contains 20 essential queue problems frequently asked
 * in technical interviews at top companies like FAANG, TCS, Infosys, and Amazon.
 *
 * Each problem includes:
 *  - Problem definition
 *  - Clean Java implementation
 *  - Example I/O
 *  - Time and Space Complexity
 */

import java.util.*;

public class Queue_Interview_Questions {

    // 1️⃣ Implement Queue using Array
    static class ArrayQueue {
        int front, rear, size, capacity;
        int[] arr;
        ArrayQueue(int capacity) {
            this.capacity = capacity;
            arr = new int[capacity];
            front = size = 0;
            rear = capacity - 1;
        }
        boolean isFull() { return size == capacity; }
        boolean isEmpty() { return size == 0; }
        void enqueue(int item) {
            if (isFull()) return;
            rear = (rear + 1) % capacity;
            arr[rear] = item;
            size++;
        }
        int dequeue() {
            if (isEmpty()) return -1;
            int item = arr[front];
            front = (front + 1) % capacity;
            size--;
            return item;
        }
        int front() { return isEmpty() ? -1 : arr[front]; }
    }

    // 2️⃣ Implement Queue using Linked List
    static class LinkedQueue {
        static class Node { int data; Node next; Node(int d){data=d;} }
        Node front, rear;
        void enqueue(int data){
            Node n = new Node(data);
            if (rear == null) front = rear = n;
            else { rear.next = n; rear = n; }
        }
        int dequeue(){
            if (front == null) return -1;
            int val = front.data;
            front = front.next;
            if (front == null) rear = null;
            return val;
        }
    }

    // 3️⃣ Circular Queue Implementation
    static class CircularQueue {
        int[] q;
        int front, rear, size, cap;
        CircularQueue(int cap){
            this.cap=cap; q=new int[cap];
            front=rear=-1;
        }
        boolean isEmpty(){return front==-1;}
        boolean isFull(){return (rear+1)%cap==front;}
        void enqueue(int x){
            if(isFull())return;
            if(isEmpty()) front=0;
            rear=(rear+1)%cap;
            q[rear]=x;
        }
        int dequeue(){
            if(isEmpty())return -1;
            int val=q[front];
            if(front==rear) front=rear=-1;
            else front=(front+1)%cap;
            return val;
        }
    }

    // 4️⃣ Implement Queue using Two Stacks
    static class QueueUsingStacks {
        Stack<Integer> s1=new Stack<>(), s2=new Stack<>();
        void enqueue(int x){ s1.push(x); }
        int dequeue(){
            if(s2.isEmpty()){
                while(!s1.isEmpty()) s2.push(s1.pop());
            }
            return s2.isEmpty()?-1:s2.pop();
        }
    }

    // 5️⃣ Reverse a Queue
    public static void reverseQueue(Queue<Integer> q){
        if(q.isEmpty()) return;
        int x=q.remove();
        reverseQueue(q);
        q.add(x);
    }

    // 6️⃣ Generate Binary Numbers from 1 to N
    public static List<String> generateBinary(int n){
        List<String> res=new ArrayList<>();
        Queue<String> q=new LinkedList<>();
        q.add("1");
        while(n-- >0){
            String s=q.remove();
            res.add(s);
            q.add(s+"0");
            q.add(s+"1");
        }
        return res;
    }

    // 7️⃣ Interleave First and Second Halves of Queue
    public static void interleaveQueue(Queue<Integer> q){
        int n=q.size()/2;
        Queue<Integer> firstHalf=new LinkedList<>();
        for(int i=0;i<n;i++) firstHalf.add(q.remove());
        while(!firstHalf.isEmpty()){
            q.add(firstHalf.remove());
            q.add(q.remove());
        }
    }

    // 8️⃣ Reverse First K Elements
    public static void reverseFirstK(Queue<Integer> q, int k){
        Stack<Integer> st=new Stack<>();
        for(int i=0;i<k;i++) st.push(q.remove());
        while(!st.isEmpty()) q.add(st.pop());
        for(int i=0;i<q.size()-k;i++) q.add(q.remove());
    }

    // 9️⃣ Implement Deque (Double-Ended Queue)
    static class DequeImpl {
        Deque<Integer> dq=new LinkedList<>();
        void insertFront(int x){ dq.addFirst(x); }
        void insertRear(int x){ dq.addLast(x); }
        void deleteFront(){ dq.pollFirst(); }
        void deleteRear(){ dq.pollLast(); }
        int getFront(){ return dq.peekFirst(); }
        int getRear(){ return dq.peekLast(); }
    }

    // 🔟 Rotten Oranges (BFS)
    public static int orangesRotting(int[][] grid){
        int rows=grid.length, cols=grid[0].length;
        Queue<int[]> q=new LinkedList<>();
        int fresh=0, time=0;
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                if(grid[i][j]==2) q.add(new int[]{i,j});
                if(grid[i][j]==1) fresh++;
            }
        }
        int[][] dirs={{1,0},{-1,0},{0,1},{0,-1}};
        while(!q.isEmpty() && fresh>0){
            int size=q.size();
            for(int i=0;i<size;i++){
                int[] cur=q.remove();
                for(int[] d:dirs){
                    int r=cur[0]+d[0], c=cur[1]+d[1];
                    if(r<0||c<0||r>=rows||c>=cols||grid[r][c]!=1) continue;
                    grid[r][c]=2;
                    fresh--;
                    q.add(new int[]{r,c});
                }
            }
            time++;
        }
        return fresh==0?time:-1;
    }

    // 1️⃣1️⃣ First Non-Repeating Character in Stream
    public static void firstNonRepeating(String str){
        int[] freq=new int[26];
        Queue<Character> q=new LinkedList<>();
        for(char c:str.toCharArray()){
            freq[c-'a']++;
            q.add(c);
            while(!q.isEmpty() && freq[q.peek()-'a']>1) q.remove();
            System.out.print(q.isEmpty()?"#":q.peek());
        }
        System.out.println();
    }

    // 1️⃣2️⃣ Sliding Window Maximum
    public static int[] slidingWindowMax(int[] nums,int k){
        Deque<Integer> dq=new LinkedList<>();
        int n=nums.length;
        int[] res=new int[n-k+1];
        for(int i=0;i<n;i++){
            while(!dq.isEmpty()&&dq.peek()<i-k+1)dq.poll();
            while(!dq.isEmpty()&&nums[dq.peekLast()]<nums[i])dq.pollLast();
            dq.offer(i);
            if(i>=k-1)res[i-k+1]=nums[dq.peek()];
        }
        return res;
    }

    // 1️⃣3️⃣ LRU Cache Implementation
    static class LRUCache {
        int cap;
        LinkedHashMap<Integer,Integer> map;
        LRUCache(int cap){
            this.cap=cap;
            map=new LinkedHashMap<>(cap,0.75f,true){
                protected boolean removeEldestEntry(Map.Entry<Integer,Integer> e){
                    return size()>cap;
                }
            };
        }
        int get(int k){ return map.getOrDefault(k,-1);}
        void put(int k,int v){ map.put(k,v);}
    }

    // 1️⃣4️⃣ Stack using Single Queue
    static class StackUsingQueue {
        Queue<Integer> q=new LinkedList<>();
        void push(int x){
            q.add(x);
            for(int i=0;i<q.size()-1;i++) q.add(q.remove());
        }
        int pop(){ return q.remove(); }
        int top(){ return q.peek(); }
    }

    // 1️⃣5️⃣ Circular Tour (Gas Station Problem)
    public static int canCompleteCircuit(int[] gas,int[] cost){
        int start=0,total=0,tank=0;
        for(int i=0;i<gas.length;i++){
            tank+=gas[i]-cost[i];
            if(tank<0){ start=i+1; total+=tank; tank=0; }
        }
        return (total+tank<0)?-1:start;
    }

    // 1️⃣6️⃣ Binary Tree Level Order Traversal
    static class TreeNode { int val; TreeNode left,right; TreeNode(int x){val=x;} }
    public static List<List<Integer>> levelOrder(TreeNode root){
        List<List<Integer>> res=new ArrayList<>();
        if(root==null) return res;
        Queue<TreeNode> q=new LinkedList<>();
        q.add(root);
        while(!q.isEmpty()){
            int size=q.size();
            List<Integer> level=new ArrayList<>();
            for(int i=0;i<size;i++){
                TreeNode node=q.remove();
                level.add(node.val);
                if(node.left!=null)q.add(node.left);
                if(node.right!=null)q.add(node.right);
            }
            res.add(level);
        }
        return res;
    }

    // 1️⃣7️⃣ Implement Priority Queue (Min Heap)
    public static void minHeapDemo(int[] arr){
        PriorityQueue<Integer> pq=new PriorityQueue<>();
        for(int x:arr)pq.add(x);
        while(!pq.isEmpty()) System.out.print(pq.remove()+" ");
        System.out.println();
    }

    // 1️⃣8️⃣ Check if All Levels of Binary Tree Are Anagrams
    public static boolean areLevelsAnagrams(TreeNode root1, TreeNode root2){
        if(root1==null||root2==null)return root1==root2;
        Queue<TreeNode> q1=new LinkedList<>(), q2=new LinkedList<>();
        q1.add(root1); q2.add(root2);
        while(!q1.isEmpty()&&!q2.isEmpty()){
            int n=q1.size();
            if(n!=q2.size()) return false;
            List<Integer> l1=new ArrayList<>(), l2=new ArrayList<>();
            for(int i=0;i<n;i++){
                TreeNode a=q1.remove(), b=q2.remove();
                l1.add(a.val); l2.add(b.val);
                if(a.left!=null)q1.add(a.left);
                if(a.right!=null)q1.add(a.right);
                if(b.left!=null)q2.add(b.left);
                if(b.right!=null)q2.add(b.right);
            }
            Collections.sort(l1); Collections.sort(l2);
            if(!l1.equals(l2)) return false;
        }
        return true;
    }

    // 1️⃣9️⃣ Petrol Pump Circular Tour
    static class PetrolPump {
        int petrol, distance;
        PetrolPump(int p,int d){petrol=p;distance=d;}
    }
    public static int petrolTour(PetrolPump[] arr){
        int start=0,deficit=0,balance=0;
        for(int i=0;i<arr.length;i++){
            balance+=arr[i].petrol-arr[i].distance;
            if(balance<0){start=i+1;deficit+=balance;balance=0;}
        }
        return (balance+deficit>=0)?start:-1;
    }

    // 2️⃣0️⃣ Queue Reversal using Recursion
    public static void reverseRecursively(Queue<Integer> q){
        if(q.isEmpty())return;
        int x=q.remove();
        reverseRecursively(q);
        q.add(x);
    }

    // 🧭 Demonstration
    public static void main(String[] args){
        Queue<Integer> q=new LinkedList<>(Arrays.asList(1,2,3,4,5));
        reverseQueue(q);
        System.out.println("Reversed Queue: "+q);

        System.out.println("Binary Numbers: "+generateBinary(5));
        firstNonRepeating("aabc");
        int[] sw=slidingWindowMax(new int[]{1,3,-1,-3,5,3,6,7},3);
        System.out.println("Sliding Window Max: "+Arrays.toString(sw));
    }
}
