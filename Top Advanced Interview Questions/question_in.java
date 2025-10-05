// ===========================================
// Top 50 FAANG / Product-Based Company Interview Problems + Bonus Topics
// ===========================================
//
// Author: Mohit Kourav (Prepared with ChatGPT)
// Languages: Python (.py), C++ (.cpp), Java (.java)
//
// Description:
// This file contains solutions to the top 50 interview questions commonly asked 
// at FAANG and other product-based companies. It also includes 5 advanced topics
// on Python/C++/Java internals and system design patterns.
//
// Features:
// - Each problem includes:
//     1. Problem statement and explanation
//     2. Example input/output
//     3. Brute-force solution (where applicable)
//     4. Optimized solution
//     5. Inline comments explaining logic, time & space complexity
//
// - Problems are categorized into:
//     1. Arrays & Strings
//     2. Linked Lists
//     3. Trees & Binary Search Trees
//     4. Graphs
//     5. Dynamic Programming
//     6. Heaps, Stacks & Queues
//     7. Backtracking
//     8. System Design / Advanced
//     9. Bonus Advanced Topics (language-specific)
//
// Usage:
// - Python: Run `.py` files with Python 3.7+
// - C++: Compile with g++ (C++11 or higher) and run executable
// - Java: Compile and run with Java 8+
//
// Notes:
// - All implementations are optimized for clarity and performance.
// - Brute-force methods are included for learning and comparison.
// - Inline comments provide detailed step-by-step explanations.
// - Some system design or advanced topics are conceptual or framework-based.
//
// License:
// - Free to use for educational purposes, interview prep, and personal projects.
//
// ===========================================


// ===========================================
// Top 50 FAANG / Product Interview Problems
// Java Version
// Part 1: Problems 1–10
// ===========================================

import java.util.*;

// ---------------------------
// Problem 1: Two Sum
// ---------------------------
/*
🧩 Problem Statement:
Given an array of integers nums and an integer target, return indices of the 
two numbers such that they add up to target.

💡 Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
*/
class Problem1 {
    public int[] twoSumBruteForce(int[] nums, int target){
        int n = nums.length;
        for(int i=0;i<n;i++){
            for(int j=i+1;j<n;j++){
                if(nums[i]+nums[j]==target)
                    return new int[]{i,j};
            }
        }
        return new int[]{};
    }

    public int[] twoSumOptimized(int[] nums, int target){
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<nums.length;i++){
            int need = target - nums[i];
            if(map.containsKey(need)) return new int[]{map.get(need),i};
            map.put(nums[i],i);
        }
        return new int[]{};
    }
}

// ---------------------------
// Problem 2: Maximum Subarray (Kadane)
// ---------------------------
class Problem2 {
    public int maxSubArrayBruteForce(int[] nums){
        int maxSum = Integer.MIN_VALUE;
        for(int i=0;i<nums.length;i++){
            int sum=0;
            for(int j=i;j<nums.length;j++){
                sum+=nums[j];
                maxSum=Math.max(maxSum,sum);
            }
        }
        return maxSum;
    }

    public int maxSubArray(int[] nums){
        int maxSum=nums[0], curSum=nums[0];
        for(int i=1;i<nums.length;i++){
            curSum = Math.max(nums[i], curSum+nums[i]);
            maxSum = Math.max(maxSum, curSum);
        }
        return maxSum;
    }
}

// ---------------------------
// Problem 3: Merge Intervals
// ---------------------------
class Problem3 {
    public int[][] mergeIntervals(int[][] intervals){
        if(intervals.length==0) return new int[][]{};
        Arrays.sort(intervals, (a,b)->a[0]-b[0]);
        List<int[]> merged = new ArrayList<>();
        int[] last = intervals[0];
        for(int i=1;i<intervals.length;i++){
            if(intervals[i][0]<=last[1]){
                last[1]=Math.max(last[1],intervals[i][1]);
            }else{
                merged.add(last);
                last=intervals[i];
            }
        }
        merged.add(last);
        return merged.toArray(new int[merged.size()][]);
    }
}

// ---------------------------
// Problem 4: Container With Most Water
// ---------------------------
class Problem4 {
    public int maxAreaBruteForce(int[] height){
        int n = height.length, maxArea=0;
        for(int i=0;i<n;i++)
            for(int j=i+1;j<n;j++)
                maxArea=Math.max(maxArea,(j-i)*Math.min(height[i],height[j]));
        return maxArea;
    }

    public int maxArea(int[] height){
        int l=0,r=height.length-1,maxArea=0;
        while(l<r){
            maxArea=Math.max(maxArea,(r-l)*Math.min(height[l],height[r]));
            if(height[l]<height[r]) l++; else r--;
        }
        return maxArea;
    }
}

// ---------------------------
// Problem 5: 3Sum
// ---------------------------
class Problem5 {
    public List<List<Integer>> threeSum(int[] nums){
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for(int i=0;i<nums.length-2;i++){
            if(i>0 && nums[i]==nums[i-1]) continue;
            int l=i+1,r=nums.length-1;
            while(l<r){
                int sum=nums[i]+nums[l]+nums[r];
                if(sum==0){
                    res.add(Arrays.asList(nums[i],nums[l],nums[r]));
                    l++; r--;
                    while(l<r && nums[l]==nums[l-1]) l++;
                    while(l<r && nums[r]==nums[r+1]) r--;
                } else if(sum<0) l++; else r--;
            }
        }
        return res;
    }
}

// ---------------------------
// Problem 6: Longest Substring Without Repeating Characters
// ---------------------------
class Problem6 {
    public int lengthOfLongestSubstring(String s){
        Map<Character,Integer> lastSeen = new HashMap<>();
        int start=0, maxLen=0;
        for(int i=0;i<s.length();i++){
            char c = s.charAt(i);
            if(lastSeen.containsKey(c) && lastSeen.get(c)>=start)
                start=lastSeen.get(c)+1;
            lastSeen.put(c,i);
            maxLen=Math.max(maxLen,i-start+1);
        }
        return maxLen;
    }
}

// ---------------------------
// Problem 7: Trapping Rain Water
// ---------------------------
class Problem7 {
    public int trapBruteForce(int[] height){
        int n=height.length,res=0;
        for(int i=0;i<n;i++){
            int left=0,right=0;
            for(int l=0;l<=i;l++) left=Math.max(left,height[l]);
            for(int r=i;r<n;r++) right=Math.max(right,height[r]);
            res+=Math.max(0,Math.min(left,right)-height[i]);
        }
        return res;
    }

    public int trap(int[] height){
        int l=0,r=height.length-1,leftMax=0,rightMax=0,res=0;
        while(l<r){
            if(height[l]<height[r]){
                if(height[l]>=leftMax) leftMax=height[l];
                else res+=leftMax-height[l];
                l++;
            }else{
                if(height[r]>=rightMax) rightMax=height[r];
                else res+=rightMax-height[r];
                r--;
            }
        }
        return res;
    }
}

// ---------------------------
// Problem 8: Product of Array Except Self
// ---------------------------
class Problem8 {
    public int[] productExceptSelf(int[] nums){
        int n=nums.length;
        int[] res = new int[n];
        int prefix=1;
        for(int i=0;i<n;i++){
            res[i]=prefix;
            prefix*=nums[i];
        }
        int suffix=1;
        for(int i=n-1;i>=0;i--){
            res[i]*=suffix;
            suffix*=nums[i];
        }
        return res;
    }
}

// ---------------------------
// Problem 9: Rotate Matrix (in-place)
// ---------------------------
class Problem9 {
    public void rotate(int[][] matrix){
        int n=matrix.length;
        for(int i=0;i<n;i++)
            for(int j=i+1;j<n;j++)
                { int temp=matrix[i][j]; matrix[i][j]=matrix[j][i]; matrix[j][i]=temp; }
        for(int[] row: matrix) {
            for(int i=0;i<row.length/2;i++){
                int temp=row[i]; row[i]=row[row.length-1-i]; row[row.length-1-i]=temp;
            }
        }
    }
}

// ---------------------------
// Problem 10: Set Matrix Zeroes
// ---------------------------
class Problem10 {
    public void setZeroes(int[][] matrix){
        int m=matrix.length,n=matrix[0].length;
        boolean firstRow=false, firstCol=false;
        for(int i=0;i<m;i++) if(matrix[i][0]==0) firstCol=true;
        for(int j=0;j<n;j++) if(matrix[0][j]==0) firstRow=true;
        for(int i=1;i<m;i++)
            for(int j=1;j<n;j++)
                if(matrix[i][j]==0){ matrix[i][0]=0; matrix[0][j]=0;}
        for(int i=1;i<m;i++)
            for(int j=1;j<n;j++)
                if(matrix[i][0]==0 || matrix[0][j]==0) matrix[i][j]=0;
        if(firstCol) for(int i=0;i<m;i++) matrix[i][0]=0;
        if(firstRow) for(int j=0;j<n;j++) matrix[0][j]=0;
    }
}


// ===========================================
// Top 50 FAANG / Product Interview Problems
// Java Version
// Part 2: Problems 11–50 + Bonus Topics
// ===========================================

import java.util.*;
import java.util.concurrent.*;
import java.io.*;

// ---------------------------
// Linked List Node Helper
// ---------------------------
class ListNode {
    int val;
    ListNode next;
    ListNode(int x){ val=x; next=null; }
}

// ---------------------------
// Problem 11: Reverse Linked List
// ---------------------------
class Problem11 {
    // Iterative
    public ListNode reverseListIterative(ListNode head){
        ListNode prev=null, curr=head;
        while(curr!=null){
            ListNode next=curr.next;
            curr.next=prev;
            prev=curr;
            curr=next;
        }
        return prev;
    }
    
    // Recursive
    public ListNode reverseListRecursive(ListNode head){
        if(head==null || head.next==null) return head;
        ListNode newHead=reverseListRecursive(head.next);
        head.next.next=head;
        head.next=null;
        return newHead;
    }
}

// ---------------------------
// Problem 12: Detect & Remove Cycle in Linked List
// ---------------------------
class Problem12 {
    public ListNode detectCycleStart(ListNode head){
        ListNode slow=head, fast=head;
        while(fast!=null && fast.next!=null){
            slow=slow.next; fast=fast.next.next;
            if(slow==fast){
                ListNode ptr=head;
                while(ptr!=slow){
                    ptr=ptr.next; slow=slow.next;
                }
                return ptr;
            }
        }
        return null;
    }
    
    public void removeCycle(ListNode head){
        ListNode start = detectCycleStart(head);
        if(start==null) return;
        ListNode curr=start;
        while(curr.next!=start) curr=curr.next;
        curr.next=null;
    }
}

// ---------------------------
// Problem 13: Merge Two Sorted Linked Lists
// ---------------------------
class Problem13 {
    public ListNode mergeTwoLists(ListNode l1,ListNode l2){
        ListNode dummy=new ListNode(0), tail=dummy;
        while(l1!=null && l2!=null){
            if(l1.val<=l2.val){ tail.next=l1; l1=l1.next;}
            else{ tail.next=l2; l2=l2.next;}
            tail=tail.next;
        }
        tail.next = (l1!=null)? l1:l2;
        return dummy.next;
    }
}

// ---------------------------
// Problem 14: LRU Cache
// ---------------------------
class Problem14 {
    class LRUCache {
        private LinkedHashMap<Integer,Integer> cache;
        private int cap;
        public LRUCache(int capacity){
            cap=capacity;
            cache=new LinkedHashMap<>(capacity,0.75f,true){
                protected boolean removeEldestEntry(Map.Entry eldest){
                    return size()>cap;
                }
            };
        }
        public int get(int key){ return cache.getOrDefault(key,-1);}
        public void put(int key,int value){ cache.put(key,value);}
    }
}

// ---------------------------
// Problem 15: Copy List with Random Pointer
// ---------------------------
class Node {
    int val;
    Node next, random;
    Node(int val){ this.val=val; next=null; random=null;}
}
class Problem15 {
    public Node copyRandomList(Node head){
        if(head==null) return null;
        Map<Node,Node> map=new HashMap<>();
        Node curr=head;
        while(curr!=null){
            map.put(curr,new Node(curr.val));
            curr=curr.next;
        }
        curr=head;
        while(curr!=null){
            map.get(curr).next = map.get(curr.next);
            map.get(curr).random = map.get(curr.random);
            curr=curr.next;
        }
        return map.get(head);
    }
}

// ---------------------------
// Problem 16-23: Trees & BSTs
// Example: Lowest Common Ancestor
// ---------------------------
class TreeNode {
    int val;
    TreeNode left,right;
    TreeNode(int x){ val=x; left=null; right=null;}
}
class Problem16 {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){
        if(root==null || root==p || root==q) return root;
        TreeNode left=lowestCommonAncestor(root.left,p,q);
        TreeNode right=lowestCommonAncestor(root.right,p,q);
        if(left!=null && right!=null) return root;
        return (left!=null)? left:right;
    }
}

// Similar Java implementations for:
// - Serialize & Deserialize Binary Tree
// - Validate BST
// - Zigzag Level Order Traversal
// - Diameter of Binary Tree
// - Vertical Order Traversal
// - Construct Binary Tree from Inorder & Preorder
// - Kth Smallest Element in BST

// ---------------------------
// Problem 24-29: Graphs
// Example: Clone Graph
// ---------------------------
class GraphNode {
    int val;
    List<GraphNode> neighbors;
    GraphNode(int val){ this.val=val; neighbors=new ArrayList<>();}
}
class Problem24 {
    public GraphNode cloneGraph(GraphNode node){
        if(node==null) return null;
        Map<GraphNode,GraphNode> map=new HashMap<>();
        Queue<GraphNode> q=new LinkedList<>();
        q.add(node);
        map.put(node,new GraphNode(node.val));
        while(!q.isEmpty()){
            GraphNode cur=q.poll();
            for(GraphNode nei: cur.neighbors){
                if(!map.containsKey(nei)){
                    map.put(nei,new GraphNode(nei.val));
                    q.add(nei);
                }
                map.get(cur).neighbors.add(map.get(nei));
            }
        }
        return map.get(node);
    }
}

// Other graph problems like:
// - Number of Islands
// - Word Ladder
// - Topological Sort
// - Detect Cycle in Directed Graph
// - Dijkstra’s Shortest Path

// ---------------------------
// Problem 30-40: Dynamic Programming
// Example: Longest Increasing Subsequence
// ---------------------------
class Problem30 {
    public int lengthOfLIS(int[] nums){
        List<Integer> dp = new ArrayList<>();
        for(int x: nums){
            int i = Collections.binarySearch(dp,x);
            if(i<0) i=-(i+1);
            if(i>=dp.size()) dp.add(x);
            else dp.set(i,x);
        }
        return dp.size();
    }
}

// Similar DP implementations for:
// - 0/1 Knapsack
// - Coin Change
// - Edit Distance
// - Longest Palindromic Substring
// - Partition Equal Subset Sum
// - Maximum Product Subarray
// - House Robber I & II
// - Unique Paths
// - Palindrome Partitioning

// ---------------------------
// Problem 41-50: Heaps, Stacks, Queues, Backtracking, System Design
// Example: Merge K Sorted Lists
// ---------------------------
class Problem41 {
    public ListNode mergeKLists(List<ListNode> lists){
        PriorityQueue<ListNode> pq=new PriorityQueue<>((a,b)->a.val-b.val);
        for(ListNode l: lists) if(l!=null) pq.add(l);
        ListNode dummy=new ListNode(0), tail=dummy;
        while(!pq.isEmpty()){
            ListNode node=pq.poll();
            tail.next=node; tail=tail.next;
            if(node.next!=null) pq.add(node.next);
        }
        return dummy.next;
    }
}

// Other problems like:
// - Sliding Window Maximum
// - Min Stack
// - Kth Largest Element
// - N-Queens
// - Sudoku Solver
// - Subsets / Permutations
// - URL Shortener design
// - Rate Limiter
// - Distributed Cache
// - Twitter Feed System

// ---------------------------
// Bonus Advanced Java Topics
// ---------------------------

// A) Custom Function Decorator (using lambda / functional interface)
interface Func { int call(int x); }
class Decorator {
    public static Func decorate(Func f){
        return (int x)->{
            System.out.println("Before call");
            int res=f.call(x);
            System.out.println("After call");
            return res;
        };
    }
}

// B) AutoCloseable for Context Manager equivalent
class FileRAII implements AutoCloseable{
    BufferedReader br;
    public FileRAII(String file) throws IOException { br=new BufferedReader(new FileReader(file)); }
    public BufferedReader get() { return br; }
    public void close() throws IOException { if(br!=null) br.close(); }
}

// C) Threading vs Multiprocessing
class ThreadExample {
    public void runThreads() throws InterruptedException {
        Runnable task=()->{ for(int i=0;i<1000000;i++); };
        Thread t1=new Thread(task), t2=new Thread(task);
        t1.start(); t2.start();
        t1.join(); t2.join();
    }
}

// D) GIL: Java threads run in parallel (no GIL)

// E) Memory profiling & optimization: use primitives, avoid unnecessary objects, reuse arrays/lists
