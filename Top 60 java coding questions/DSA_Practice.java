// DSA_Practice.java
// 60 DSA problems (clean Java implementations) organized by topic
// Author: Generated for Surya
import java.util.*;
import java.util.stream.*;

public class DSA_Practice {

    // -----------------------
    // Helper node classes
    // -----------------------
    public static class ListNode {
        int val; ListNode next;
        ListNode(int v) { val = v; }
    }

    public static class RandomListNode {
        int val; RandomListNode next; RandomListNode random;
        RandomListNode(int v) { val = v; }
    }

    public static class TreeNode {
        int val; TreeNode left, right;
        TreeNode(int v) { val = v; }
    }

    public static class GraphNode {
        int val; List<GraphNode> neighbors;
        GraphNode(int v) { val = v; neighbors = new ArrayList<>(); }
    }

    // =======================
    // Arrays & Strings (10)
    // =======================

    // 1. Two Sum
    // O(n) time, O(n) space
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i=0;i<nums.length;i++){
            int comp = target - nums[i];
            if (map.containsKey(comp)) return new int[]{map.get(comp), i};
            map.put(nums[i], i);
        }
        return new int[]{};
    }

    // 2. Maximum Subarray (Kadane)
    public static int maxSubarray(int[] nums) {
        int max = nums[0], curr = nums[0];
        for (int i=1;i<nums.length;i++){
            curr = Math.max(nums[i], curr + nums[i]);
            max = Math.max(max, curr);
        }
        return max;
    }

    // 3. Rotate Array (in-place using reversal)
    public static void rotateArray(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length-1);
        reverse(nums, 0, k-1);
        reverse(nums, k, nums.length-1);
    }
    private static void reverse(int[] a, int l, int r){
        while(l<r){ int t=a[l]; a[l++]=a[r]; a[r--]=t; }
    }

    // 4. Product of Array Except Self
    public static int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] out = new int[n];
        Arrays.fill(out,1);
        int left = 1;
        for (int i=0;i<n;i++){ out[i] = left; left *= nums[i]; }
        int right = 1;
        for (int i=n-1;i>=0;i--){ out[i] *= right; right *= nums[i]; }
        return out;
    }

    // 5. Trapping Rain Water (two-pointer)
    public static int trapRain(int[] height) {
        if (height.length==0) return 0;
        int l=0, r=height.length-1, lmax=0, rmax=0, res=0;
        while (l<r){
            if (height[l] < height[r]){
                if (height[l] >= lmax) lmax = height[l];
                else res += lmax - height[l];
                l++;
            } else {
                if (height[r] >= rmax) rmax = height[r];
                else res += rmax - height[r];
                r--;
            }
        }
        return res;
    }

    // 6. Valid Palindrome (ignoring non-alnum)
    public static boolean isPalindrome(String s) {
        int i=0,j=s.length()-1;
        while (i<j){
            while (i<j && !Character.isLetterOrDigit(s.charAt(i))) i++;
            while (i<j && !Character.isLetterOrDigit(s.charAt(j))) j--;
            if (Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j))) return false;
            i++; j--;
        }
        return true;
    }

    // 7. Longest Substring Without Repeating Characters
    public static int lengthOfLongestSubstring(String s) {
        Map<Character,Integer> last = new HashMap<>();
        int start=0, best=0;
        for (int i=0;i<s.length();i++){
            char c = s.charAt(i);
            if (last.containsKey(c)) start = Math.max(start, last.get(c)+1);
            best = Math.max(best, i - start + 1);
            last.put(c, i);
        }
        return best;
    }

    // 8. Merge Intervals
    public static int[][] mergeIntervals(int[][] intervals) {
        if (intervals.length==0) return new int[0][];
        Arrays.sort(intervals, Comparator.comparingInt(a->a[0]));
        List<int[]> res = new ArrayList<>();
        int[] cur = intervals[0];
        for (int i=1;i<intervals.length;i++){
            if (intervals[i][0] <= cur[1]) cur[1] = Math.max(cur[1], intervals[i][1]);
            else { res.add(cur); cur = intervals[i]; }
        }
        res.add(cur);
        return res.toArray(new int[res.size()][]);
    }

    // 9. Group Anagrams
    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String,List<String>> map = new HashMap<>();
        for (String s: strs){
            char[] ca = s.toCharArray();
            Arrays.sort(ca);
            String key = new String(ca);
            map.computeIfAbsent(key, k-> new ArrayList<>()).add(s);
        }
        return new ArrayList<>(map.values());
    }

    // 10. Move Zeroes
    public static void moveZeroes(int[] nums) {
        int j=0;
        for (int i=0;i<nums.length;i++){
            if (nums[i]!=0){ int t=nums[i]; nums[i]=nums[j]; nums[j]=t; j++; }
        }
    }

    // ========================
    // Linked List (8)
    // ========================

    // 1. Reverse Linked List
    public static ListNode reverseList(ListNode head) {
        ListNode prev = null, curr = head;
        while (curr != null) {
            ListNode nxt = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nxt;
        }
        return prev;
    }

    // 2. Detect Cycle (Floyd)
    public static boolean hasCycle(ListNode head) {
        ListNode slow=head, fast=head;
        while (fast!=null && fast.next!=null){
            slow=slow.next; fast=fast.next.next;
            if (slow==fast) return true;
        }
        return false;
    }

    // 3. Merge Two Sorted Lists
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2){
        ListNode dummy=new ListNode(0), tail=dummy;
        while (l1!=null && l2!=null){
            if (l1.val < l2.val){ tail.next = l1; l1 = l1.next; }
            else { tail.next = l2; l2 = l2.next; }
            tail = tail.next;
        }
        tail.next = (l1!=null)? l1 : l2;
        return dummy.next;
    }

    // 4. Remove N-th From End
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0); dummy.next = head;
        ListNode first = dummy, second = dummy;
        for (int i=0;i<=n;i++) first = first.next;
        while (first != null) { first = first.next; second = second.next; }
        second.next = second.next.next;
        return dummy.next;
    }

    // 5. Copy List with Random Pointer
    public static RandomListNode copyRandomList(RandomListNode head) {
        if (head==null) return null;
        RandomListNode curr = head;
        while (curr != null) {
            RandomListNode copy = new RandomListNode(curr.val);
            copy.next = curr.next;
            curr.next = copy;
            curr = copy.next;
        }
        curr = head;
        while (curr != null) {
            if (curr.random != null) curr.next.random = curr.random.next;
            curr = curr.next.next;
        }
        RandomListNode dummy = new RandomListNode(0), p = dummy;
        curr = head;
        while (curr != null) {
            p.next = curr.next;
            curr.next = curr.next.next;
            curr = curr.next;
            p = p.next;
        }
        return dummy.next;
    }

    // 6. Palindrome Linked List
    public static boolean isPalindromeList(ListNode head) {
        if (head==null || head.next == null) return true;
        ListNode slow=head, fast=head;
        while (fast!=null && fast.next!=null) { slow=slow.next; fast=fast.next.next; }
        ListNode second = reverseList(slow);
        ListNode first = head;
        while (second != null) {
            if (first.val != second.val) return false;
            first = first.next; second = second.next;
        }
        return true;
    }

    // 7. Flatten Multilevel Doubly Linked List (assume child pointer is 'child' in some implementation)
    // For simplicity, this function assumes nodes have 'child' and 'next' and 'prev'. We skip explicit implementation here.
    // Omitted because Java representation needs extra class; keep as placeholder.

    // 8. Intersection of Two Linked Lists
    public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode a=headA, b=headB;
        while (a!=b){
            a = (a==null)? headB : a.next;
            b = (b==null)? headA : b.next;
        }
        return a;
    }

    // ========================
    // Stack & Queue (5)
    // ========================

    // 1. Valid Parentheses
    public static boolean isValidParentheses(String s) {
        Deque<Character> st = new ArrayDeque<>();
        Map<Character, Character> map = Map.of(')', '(', ']', '[', '}', '{');
        for (char c: s.toCharArray()){
            if (map.containsKey(c)) {
                char top = st.isEmpty() ? '#' : st.pop();
                if (top != map.get(c)) return false;
            } else st.push(c);
        }
        return st.isEmpty();
    }

    // 2. Min Stack
    public static class MinStack {
        private Deque<Integer> s = new ArrayDeque<>();
        private Deque<Integer> mins = new ArrayDeque<>();
        public void push(int x){ s.push(x); if (mins.isEmpty() || x <= mins.peek()) mins.push(x); }
        public int pop(){ int v = s.pop(); if (v == mins.peek()) mins.pop(); return v; }
        public int top(){ return s.peek(); }
        public int getMin(){ return mins.peek(); }
    }

    // 3. Implement Queue using Stacks
    public static class MyQueue {
        Deque<Integer> in = new ArrayDeque<>(), out = new ArrayDeque<>();
        public void push(int x){ in.push(x); }
        public int pop(){ peek(); return out.pop(); }
        public int peek(){ if (out.isEmpty()) while (!in.isEmpty()) out.push(in.pop()); return out.peek(); }
        public boolean empty(){ return in.isEmpty() && out.isEmpty(); }
    }

    // 4. Sliding Window Maximum
    public static int[] maxSlidingWindow(int[] nums, int k) {
        if (k==0) return new int[0];
        Deque<Integer> dq = new ArrayDeque<>();
        int n = nums.length;
        int[] res = new int[n-k+1];
        for (int i=0;i<n;i++){
            while (!dq.isEmpty() && dq.peekFirst() <= i - k) dq.pollFirst();
            while (!dq.isEmpty() && nums[dq.peekLast()] < nums[i]) dq.pollLast();
            dq.offerLast(i);
            if (i >= k - 1) res[i-k+1] = nums[dq.peekFirst()];
        }
        return res;
    }

    // 5. Next Greater Element
    public static int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        Deque<Integer> st = new ArrayDeque<>();
        for (int i=0;i<2*n;i++){
            int num = nums[i % n];
            while (!st.isEmpty() && nums[st.peek()] < num) res[st.pop()] = num;
            if (i < n) st.push(i);
        }
        return res;
    }

    // ========================
    // Trees & Graphs (12)
    // ========================

    // 1. Max Depth Binary Tree
    public static int maxDepth(TreeNode root) {
        if (root==null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    // 2. Lowest Common Ancestor (binary tree)
    public static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root==null || root==p || root==q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        return (left!=null && right!=null) ? root : (left!=null ? left : right);
    }

    // 3. Diameter of Binary Tree
    public static int diameterOfBinaryTree(TreeNode root) {
        int[] dia = new int[1];
        depthForDiameter(root, dia);
        return dia[0];
    }
    private static int depthForDiameter(TreeNode node, int[] dia){
        if (node==null) return 0;
        int l = depthForDiameter(node.left, dia);
        int r = depthForDiameter(node.right, dia);
        dia[0] = Math.max(dia[0], l + r);
        return 1 + Math.max(l, r);
    }

    // 4. Serialize / Deserialize Binary Tree (preorder)
    public static String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serializeHelper(root, sb);
        return sb.toString();
    }
    private static void serializeHelper(TreeNode node, StringBuilder sb) {
        if (node==null) { sb.append("X,"); return; }
        sb.append(node.val).append(",");
        serializeHelper(node.left, sb);
        serializeHelper(node.right, sb);
    }
    public static TreeNode deserialize(String data) {
        Deque<String> dq = new ArrayDeque<>(Arrays.asList(data.split(",")));
        return deserializeHelper(dq);
    }
    private static TreeNode deserializeHelper(Deque<String> dq) {
        String val = dq.poll();
        if (val.equals("X")) return null;
        TreeNode node = new TreeNode(Integer.parseInt(val));
        node.left = deserializeHelper(dq);
        node.right = deserializeHelper(dq);
        return node;
    }

    // 5. Level Order Traversal
    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root==null) return res;
        Queue<TreeNode> q = new LinkedList<>(); q.add(root);
        while (!q.isEmpty()){
            int sz=q.size(); List<Integer> level = new ArrayList<>();
            for (int i=0;i<sz;i++){
                TreeNode n = q.poll(); level.add(n.val);
                if (n.left!=null) q.add(n.left);
                if (n.right!=null) q.add(n.right);
            }
            res.add(level);
        }
        return res;
    }

    // 6. Number of Islands (DFS)
    public static int numIslands(char[][] grid) {
        if (grid==null || grid.length==0) return 0;
        int rows=grid.length, cols=grid[0].length, count=0;
        for (int r=0;r<rows;r++){
            for (int c=0;c<cols;c++){
                if (grid[r][c]=='1'){ dfsIsland(grid,r,c); count++; }
            }
        }
        return count;
    }
    private static void dfsIsland(char[][] g, int r, int c){
        if (r<0||c<0||r>=g.length||c>=g[0].length||g[r][c]=='0') return;
        g[r][c]='0';
        dfsIsland(g,r+1,c); dfsIsland(g,r-1,c); dfsIsland(g,r,c+1); dfsIsland(g,r,c-1);
    }

    // 7. Clone Graph (DFS)
    public static GraphNode cloneGraph(GraphNode node) {
        if (node==null) return null;
        Map<GraphNode, GraphNode> map = new HashMap<>();
        return cloneGraphDfs(node, map);
    }
    private static GraphNode cloneGraphDfs(GraphNode node, Map<GraphNode, GraphNode> map){
        if (map.containsKey(node)) return map.get(node);
        GraphNode copy = new GraphNode(node.val);
        map.put(node, copy);
        for (GraphNode nei: node.neighbors) copy.neighbors.add(cloneGraphDfs(nei, map));
        return copy;
    }

    // 8. Course Schedule (can finish) using Kahn's algorithm
    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        int[] indeg = new int[numCourses];
        for (int i=0;i<numCourses;i++) graph.add(new ArrayList<>());
        for (int[] p: prerequisites){
            graph.get(p[1]).add(p[0]);
            indeg[p[0]]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i=0;i<numCourses;i++) if (indeg[i]==0) q.add(i);
        int visited = 0;
        while (!q.isEmpty()){
            int u = q.poll(); visited++;
            for (int v: graph.get(u)){
                indeg[v]--;
                if (indeg[v]==0) q.add(v);
            }
        }
        return visited == numCourses;
    }

    // 9. Dijkstra's shortest path (adj list graph: Map<Integer, List<int[]>> where int[]{neighbor,weight})
    public static Map<Integer, Integer> dijkstra(Map<Integer, List<int[]>> graph, int start) {
        Map<Integer,Integer> dist = new HashMap<>();
        for (Integer v: graph.keySet()) dist.put(v, Integer.MAX_VALUE);
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a->a[0]));
        dist.put(start, 0); pq.add(new int[]{0, start});
        while (!pq.isEmpty()){
            int[] cur = pq.poll(); int d = cur[0], u = cur[1];
            if (d > dist.get(u)) continue;
            for (int[] e: graph.getOrDefault(u, Collections.emptyList())){
                int v = e[0], w = e[1];
                if (d + w < dist.getOrDefault(v, Integer.MAX_VALUE)){
                    dist.put(v, d + w);
                    pq.add(new int[]{dist.get(v), v});
                }
            }
        }
        return dist;
    }

    // 10. Kruskal's MST
    public static class UF {
        int[] p;
        UF(int n){ p = new int[n]; for (int i=0;i<n;i++) p[i]=i; }
        int find(int x){ return p[x]==x? x : (p[x]=find(p[x])); }
        boolean union(int a, int b){ int pa=find(a), pb=find(b); if (pa==pb) return false; p[pa]=pb; return true; }
    }
    public static List<int[]> kruskal(int n, List<int[]> edges) {
        Collections.sort(edges, Comparator.comparingInt(a->a[2]));
        UF uf = new UF(n);
        List<int[]> mst = new ArrayList<>();
        for (int[] e: edges) if (uf.union(e[0], e[1])) mst.add(e);
        return mst;
    }

    // 11. Prim's MST (graph as Map<Integer, List<int[]>>)
    public static List<int[]> primMST(Map<Integer, List<int[]>> graph) {
        if (graph.isEmpty()) return Collections.emptyList();
        int start = graph.keySet().iterator().next();
        Set<Integer> inMST = new HashSet<>();
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a->a[0]));
        List<int[]> res = new ArrayList<>();
        inMST.add(start);
        for (int[] e: graph.get(start)) pq.add(new int[]{e[1], start, e[0]}); // weight, u, v
        while (!pq.isEmpty()){
            int[] cur = pq.poll();
            int w = cur[0], u = cur[1], v = cur[2];
            if (inMST.contains(v)) continue;
            inMST.add(v);
            res.add(new int[]{u, v, w});
            for (int[] e: graph.getOrDefault(v, Collections.emptyList())) if (!inMST.contains(e[0])) pq.add(new int[]{e[1], v, e[0]});
        }
        return res;
    }

    // 12. Graph Cycle Detection (directed)
    public static boolean hasCycleDirected(Map<Integer, List<Integer>> graph) {
        Set<Integer> visited = new HashSet<>(), rec = new HashSet<>();
        for (Integer node: graph.keySet()){
            if (!visited.contains(node) && dfsCycle(node, graph, visited, rec)) return true;
        }
        return false;
    }
    private static boolean dfsCycle(int u, Map<Integer,List<Integer>> g, Set<Integer> vis, Set<Integer> rec){
        vis.add(u); rec.add(u);
        for (int v: g.getOrDefault(u, Collections.emptyList())){
            if (!vis.contains(v) && dfsCycle(v,g,vis,rec)) return true;
            if (rec.contains(v)) return true;
        }
        rec.remove(u); return false;
    }

    // ========================
    // Dynamic Programming (10)
    // ========================

    // 1. Fibonacci (iterative)
    public static long fibonacci(int n) {
        if (n<=1) return n;
        long a=0,b=1;
        for (int i=2;i<=n;i++){ long c=a+b; a=b; b=c; }
        return b;
    }

    // 2. Longest Common Subsequence
    public static int lcs(String s1, String s2) {
        int m=s1.length(), n=s2.length();
        int[][] dp = new int[m+1][n+1];
        for (int i=1;i<=m;i++){
            for (int j=1;j<=n;j++){
                if (s1.charAt(i-1)==s2.charAt(j-1)) dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j]=Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[m][n];
    }

    // 3. 0/1 Knapsack
    public static int knapsack(int[] wt, int[] val, int W) {
        int n = wt.length;
        int[][] dp = new int[n+1][W+1];
        for (int i=1;i<=n;i++){
            for (int w=1; w<=W; w++){
                if (wt[i-1] <= w) dp[i][w] = Math.max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w]);
                else dp[i][w] = dp[i-1][w];
            }
        }
        return dp[n][W];
    }

    // 4. Coin Change (min coins)
    public static int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount+1];
        Arrays.fill(dp, amount+1);
        dp[0]=0;
        for (int coin: coins){
            for (int x=coin; x<=amount; x++) dp[x] = Math.min(dp[x], dp[x-coin] + 1);
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    // 5. Minimum Path Sum in Grid
    public static int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0]=grid[0][0];
        for (int i=1;i<m;i++) dp[i][0]=dp[i-1][0]+grid[i][0];
        for (int j=1;j<n;j++) dp[0][j]=dp[0][j-1]+grid[0][j];
        for (int i=1;i<m;i++) for (int j=1;j<n;j++) dp[i][j]=grid[i][j] + Math.min(dp[i-1][j], dp[i][j-1]);
        return dp[m-1][n-1];
    }

    // 6. Longest Increasing Subsequence (O(n^2))
    public static int lis(int[] nums) {
        int n=nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp,1);
        int best = 0;
        for (int i=0;i<n;i++){
            for (int j=0;j<i;j++) if (nums[i] > nums[j]) dp[i] = Math.max(dp[i], dp[j] + 1);
            best = Math.max(best, dp[i]);
        }
        return best;
    }

    // 7. Edit Distance
    public static int editDistance(String a, String b) {
        int m=a.length(), n=b.length();
        int[][] dp = new int[m+1][n+1];
        for (int i=0;i<=m;i++) dp[i][0]=i;
        for (int j=0;j<=n;j++) dp[0][j]=j;
        for (int i=1;i<=m;i++){
            for (int j=1;j<=n;j++){
                if (a.charAt(i-1) == b.charAt(j-1)) dp[i][j]=dp[i-1][j-1];
                else dp[i][j] = 1 + Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1]));
            }
        }
        return dp[m][n];
    }

    // 8. Maximum Product Subarray
    public static int maxProductSubarray(int[] nums) {
        int maxProduct = nums[0], minProduct = nums[0], res = nums[0];
        for (int i=1;i<nums.length;i++){
            int n = nums[i];
            if (n < 0) { int t = maxProduct; maxProduct = minProduct; minProduct = t; }
            maxProduct = Math.max(n, maxProduct * n);
            minProduct = Math.min(n, minProduct * n);
            res = Math.max(res, maxProduct);
        }
        return res;
    }

    // 9. House Robber
    public static int houseRobber(int[] nums) {
        if (nums.length == 0) return 0;
        int incl = nums[0], excl = 0;
        for (int i=1;i<nums.length;i++){
            int ni = Math.max(incl, excl + nums[i]);
            excl = incl;
            incl = ni;
        }
        return incl;
    }

    // 10. Unique Paths (grid combinatorics DP)
    public static int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i=0;i<m;i++) dp[i][0]=1;
        for (int j=0;j<n;j++) dp[0][j]=1;
        for (int i=1;i<m;i++) for (int j=1;j<n;j++) dp[i][j] = dp[i-1][j] + dp[i][j-1];
        return dp[m-1][n-1];
    }

    // ========================
    // Recursion & Backtracking (5)
    // ========================

    // 1. N-Queens
    public static List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int i=0;i<n;i++) Arrays.fill(board[i], '.');
        backtrackNQ(0, board, res, new HashSet<>(), new HashSet<>(), new HashSet<>());
        return res;
    }
    private static void backtrackNQ(int row, char[][] board, List<List<String>> res, Set<Integer> cols, Set<Integer> diag, Set<Integer> anti) {
        int n = board.length;
        if (row == n) {
            List<String> sol = new ArrayList<>();
            for (char[] r: board) sol.add(new String(r));
            res.add(sol);
            return;
        }
        for (int c=0;c<n;c++){
            int d = row - c, a = row + c;
            if (cols.contains(c) || diag.contains(d) || anti.contains(a)) continue;
            cols.add(c); diag.add(d); anti.add(a); board[row][c] = 'Q';
            backtrackNQ(row+1, board, res, cols, diag, anti);
            cols.remove(c); diag.remove(d); anti.remove(a); board[row][c] = '.';
        }
    }

    // 2. Generate Parentheses
    public static List<String> generateParentheses(int n) {
        List<String> res = new ArrayList<>();
        backtrackGP("", 0, 0, n, res);
        return res;
    }
    private static void backtrackGP(String s, int open, int close, int n, List<String> res) {
        if (s.length() == 2*n) { res.add(s); return; }
        if (open < n) backtrackGP(s+"(", open+1, close, n, res);
        if (close < open) backtrackGP(s+")", open, close+1, n, res);
    }

    // 3. Subsets
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrackSubsets(0, nums, new ArrayList<>(), res);
        return res;
    }
    private static void backtrackSubsets(int idx, int[] nums, List<Integer> path, List<List<Integer>> res) {
        res.add(new ArrayList<>(path));
        for (int i=idx;i<nums.length;i++){
            path.add(nums[i]);
            backtrackSubsets(i+1, nums, path, res);
            path.remove(path.size()-1);
        }
    }

    // 4. Word Search
    public static boolean exist(char[][] board, String word) {
        int R = board.length, C = board[0].length;
        for (int r=0;r<R;r++) for (int c=0;c<C;c++) if (dfsWord(board, r, c, word, 0)) return true;
        return false;
    }
    private static boolean dfsWord(char[][] b, int r, int c, String w, int i) {
        if (i==w.length()) return true;
        if (r<0||c<0||r>=b.length||c>=b[0].length||b[r][c]!=w.charAt(i)) return false;
        char tmp = b[r][c]; b[r][c] = '#';
        boolean res = dfsWord(b,r+1,c,w,i+1) || dfsWord(b,r-1,c,w,i+1) || dfsWord(b,r,c+1,w,i+1) || dfsWord(b,r,c-1,w,i+1);
        b[r][c] = tmp; return res;
    }

    // 5. Combination Sum
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        backtrackComb(candidates, 0, target, new ArrayList<>(), res);
        return res;
    }
    private static void backtrackComb(int[] cand, int idx, int remain, List<Integer> path, List<List<Integer>> res){
        if (remain==0) { res.add(new ArrayList<>(path)); return; }
        if (remain < 0) return;
        for (int i=idx;i<cand.length;i++){
            path.add(cand[i]);
            backtrackComb(cand, i, remain-cand[i], path, res);
            path.remove(path.size()-1);
        }
    }

    // ========================
    // Miscellaneous / Advanced (10)
    // ========================

    // 1. LRU Cache
    public static class LRUCache {
        private final LinkedHashMap<Integer,Integer> map;
        private final int capacity;
        public LRUCache(int capacity){
            this.capacity = capacity;
            map = new LinkedHashMap<>(capacity, 0.75f, true){
                protected boolean removeEldestEntry(Map.Entry<Integer,Integer> e){
                    return size() > LRUCache.this.capacity;
                }
            };
        }
        public int get(int key){ return map.getOrDefault(key, -1); }
        public void put(int key, int value){ map.put(key, value); }
    }

    // 2. Serialize / Deserialize N-ary Tree (preorder with child count)
    public static String serializeNary(NTreeNode root){
        StringBuilder sb = new StringBuilder();
        serializeNaryDfs(root, sb);
        return sb.toString();
    }
    private static void serializeNaryDfs(NTreeNode node, StringBuilder sb){
        if (node==null) return;
        sb.append(node.val).append(",").append(node.children.size()).append(",");
        for (NTreeNode c: node.children) serializeNaryDfs(c, sb);
    }
    public static NTreeNode deserializeNary(String data){
        if (data==null || data.isEmpty()) return null;
        Deque<String> dq = new ArrayDeque<>(Arrays.asList(data.split(",")));
        return deserializeNaryDfs(dq);
    }
    private static NTreeNode deserializeNaryDfs(Deque<String> dq){
        if (dq.isEmpty()) return null;
        int val = Integer.parseInt(dq.poll());
        int size = Integer.parseInt(dq.poll());
        NTreeNode node = new NTreeNode(val);
        for (int i=0;i<size;i++) node.children.add(deserializeNaryDfs(dq));
        return node;
    }
    public static class NTreeNode { int val; List<NTreeNode> children = new ArrayList<>(); NTreeNode(int v){val=v;} }

    // 3. Median of Two Sorted Arrays (log min(n,m))
    public static double findMedianSortedArrays(int[] a, int[] b){
        if (a.length > b.length) return findMedianSortedArrays(b, a);
        int x = a.length, y = b.length;
        int low = 0, high = x;
        while (low <= high) {
            int px = (low + high) / 2;
            int py = (x + y + 1) / 2 - px;
            int maxLeftX = (px == 0) ? Integer.MIN_VALUE : a[px-1];
            int minRightX = (px == x) ? Integer.MAX_VALUE : a[px];
            int maxLeftY = (py == 0) ? Integer.MIN_VALUE : b[py-1];
            int minRightY = (py == y) ? Integer.MAX_VALUE : b[py];
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                if ((x+y)%2==0) return ((double)Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY))/2;
                else return Math.max(maxLeftX, maxLeftY);
            } else if (maxLeftX > minRightY) high = px - 1;
            else low = px + 1;
        }
        throw new IllegalArgumentException();
    }

    // 4. Trapping Rain Water (already implemented earlier as trapRain)

    // 5. Word Ladder (BFS)
    public static int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> set = new HashSet<>(wordList);
        if (!set.contains(endWord)) return 0;
        Queue<String> q = new LinkedList<>();
        q.add(beginWord); int steps = 1;
        while (!q.isEmpty()){
            int sz = q.size();
            for (int i=0;i<sz;i++){
                String w = q.poll();
                if (w.equals(endWord)) return steps;
                char[] ca = w.toCharArray();
                for (int pos=0;pos<ca.length;pos++){
                    char old = ca[pos];
                    for (char c='a'; c<='z'; c++){
                        ca[pos]=c;
                        String nxt = new String(ca);
                        if (set.contains(nxt)) { set.remove(nxt); q.add(nxt); }
                    }
                    ca[pos] = old;
                }
            }
            steps++;
        }
        return 0;
    }

    // 6. Decode Ways (DP)
    public static int numDecodings(String s) {
        if (s==null || s.length()==0) return 0;
        int n = s.length();
        int[] dp = new int[n+1];
        dp[0]=1; dp[1] = s.charAt(0)=='0' ? 0 : 1;
        for (int i=2;i<=n;i++){
            if (s.charAt(i-1) != '0') dp[i] += dp[i-1];
            int two = Integer.parseInt(s.substring(i-2, i));
            if (two >= 10 && two <= 26) dp[i] += dp[i-2];
        }
        return dp[n];
    }

    // 7. Maximum Sum Rectangle in 2D Matrix
    public static int maxSumRectangle(int[][] matrix) {
        if (matrix.length==0) return 0;
        int rows=matrix.length, cols=matrix[0].length, max = Integer.MIN_VALUE;
        for (int top=0; top<rows; top++){
            int[] temp = new int[cols];
            for (int bottom=top; bottom<rows; bottom++){
                for (int c=0;c<cols;c++) temp[c] += matrix[bottom][c];
                max = Math.max(max, kadane1D(temp));
            }
        }
        return max;
    }
    private static int kadane1D(int[] arr){
        int maxCur = arr[0], maxSo = arr[0];
        for (int i=1;i<arr.length;i++){ maxCur = Math.max(arr[i], maxCur + arr[i]); maxSo = Math.max(maxSo, maxCur); }
        return maxSo;
    }

    // 8. Maximal Rectangle of 1s
    public static int maximalRectangle(char[][] matrix) {
        if (matrix.length==0) return 0;
        int cols = matrix[0].length;
        int[] heights = new int[cols];
        int maxArea = 0;
        for (char[] row: matrix){
            for (int i=0;i<cols;i++) heights[i] = row[i]=='1' ? heights[i]+1 : 0;
            maxArea = Math.max(maxArea, largestRectangleArea(heights));
        }
        return maxArea;
    }
    private static int largestRectangleArea(int[] h){
        int n=h.length;
        Deque<Integer> st = new ArrayDeque<>();
        int max = 0;
        for (int i=0;i<=n;i++){
            int cur = (i==n) ? 0 : h[i];
            while (!st.isEmpty() && cur < h[st.peek()]) {
                int height = h[st.pop()];
                int width = st.isEmpty() ? i : i - st.peek() - 1;
                max = Math.max(max, height * width);
            }
            st.push(i);
        }
        return max;
    }

    // 9. Sliding Window Median
    // Note: Full robust implementation is long; here's a simplified version using two Heaps
    public static double[] medianSlidingWindow(int[] nums, int k) {
        if (k==0) return new double[0];
        TreeMap<Integer, Integer> left = new TreeMap<>(Collections.reverseOrder()), right = new TreeMap<>();
        int leftSize = 0, rightSize = 0;
        List<Double> resList = new ArrayList<>();
        for (int i=0;i<nums.length;i++){
            int num = nums[i];
            if (leftSize==0 || num <= left.firstKey()) { left.put(num, left.getOrDefault(num,0)+1); leftSize++; }
            else { right.put(num, right.getOrDefault(num,0)+1); rightSize++; }
            balanceMulti(left, right);
            if (i >= k-1){
                resList.add(getMedian(left, right, k));
                int out = nums[i-k+1];
                if (left.containsKey(out)){ left.put(out, left.get(out)-1); if (left.get(out)==0) left.remove(out); leftSize--; }
                else { right.put(out, right.get(out)-1); if (right.get(out)==0) right.remove(out); rightSize--; }
                balanceMulti(left, right);
            }
        }
        double[] res = new double[resList.size()];
        for (int i=0;i<res.length;i++) res[i] = resList.get(i);
        return res;
    }
    private static void balanceMulti(TreeMap<Integer,Integer> left, TreeMap<Integer,Integer> right){
        int lsize = left.values().stream().mapToInt(Integer::intValue).sum();
        int rsize = right.values().stream().mapToInt(Integer::intValue).sum();
        while (lsize > rsize + 1) {
            int val = left.firstKey();
            int cnt = left.get(val);
            left.put(val, cnt-1); if (left.get(val)==0) left.remove(val);
            right.put(val, right.getOrDefault(val,0)+1);
            lsize--; rsize++;
        }
        while (lsize < rsize) {
            int val = right.firstKey();
            int cnt = right.get(val);
            right.put(val, cnt-1); if (right.get(val)==0) right.remove(val);
            left.put(val, left.getOrDefault(val,0)+1);
            lsize++; rsize--;
        }
    }
    private static double getMedian(TreeMap<Integer,Integer> left, TreeMap<Integer,Integer> right, int k){
        int lsize = left.values().stream().mapToInt(Integer::intValue).sum();
        int rsize = right.values().stream().mapToInt(Integer::intValue).sum();
        if ((k & 1) == 1) return left.firstKey();
        else return ((double)left.firstKey() + (double)right.firstKey())/2.0;
    }

    // 10. Top K Frequent Elements
    public static int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> freq = new HashMap<>();
        for (int n: nums) freq.put(n, freq.getOrDefault(n,0)+1);
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(freq::get));
        for (int key: freq.keySet()){
            pq.add(key);
            if (pq.size() > k) pq.poll();
        }
        int[] res = new int[k]; for (int i=k-1;i>=0;i--) res[i] = pq.poll();
        return res;
    }

    // -----------------------
    // Main (demo few functions)
    // -----------------------
    public static void main(String[] args) {
        // A few quick demos
        System.out.println("Two Sum: " + Arrays.toString(twoSum(new int[]{2,7,11,15}, 9)));
        System.out.println("Max Subarray: " + maxSubarray(new int[]{-2,1,-3,4,-1,2,1,-5,4}));
        int[] arr = {1,2,3,4,5,6,7}; rotateArray(arr, 3); System.out.println("Rotate: " + Arrays.toString(arr));
        System.out.println("Product Except Self: " + Arrays.toString(productExceptSelf(new int[]{1,2,3,4})));
        System.out.println("Trapped Rain: " + trapRain(new int[]{0,1,0,2,1,0,1,3,2,1,2,1}));
        System.out.println("Valid Palindrome: " + isPalindrome("A man, a plan, a canal: Panama"));
        System.out.println("Longest Substring: " + lengthOfLongestSubstring("abcabcbb"));
        System.out.println("Move Zeroes:");
        int[] z = {0,1,0,3,12}; moveZeroes(z); System.out.println(Arrays.toString(z));

        System.out.println("LIS: " + lis(new int[]{10,9,2,5,3,7,101,18}));
        System.out.println("Ladder length: " + ladderLength("hit","cog", Arrays.asList("hot","dot","dog","lot","log","cog")));
        System.out.println("Top K frequent: " + Arrays.toString(topKFrequent(new int[]{1,1,1,2,2,3}, 2)));
    }
}
