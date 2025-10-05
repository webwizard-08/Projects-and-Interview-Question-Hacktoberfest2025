# // ===========================================
# // Top 50 FAANG / Product-Based Company Interview Problems + Bonus Topics
# // ===========================================
# //
# // Author: Mohit Kourav (Prepared with ChatGPT)
# // Languages: Python (.py), C++ (.cpp), Java (.java)
# //
# // Description:
# // This file contains solutions to the top 50 interview questions commonly asked 
# // at FAANG and other product-based companies. It also includes 5 advanced topics
# // on Python/C++/Java internals and system design patterns.
# //
# // Features:
# // - Each problem includes:
# //     1. Problem statement and explanation
# //     2. Example input/output
# //     3. Brute-force solution (where applicable)
# //     4. Optimized solution
# //     5. Inline comments explaining logic, time & space complexity
# //
# // - Problems are categorized into:
# //     1. Arrays & Strings
# //     2. Linked Lists
# //     3. Trees & Binary Search Trees
# //     4. Graphs
# //     5. Dynamic Programming
# //     6. Heaps, Stacks & Queues
# //     7. Backtracking
# //     8. System Design / Advanced
# //     9. Bonus Advanced Topics (language-specific)
# //
# // Usage:
# // - Python: Run `.py` files with Python 3.7+
# // - C++: Compile with g++ (C++11 or higher) and run executable
# // - Java: Compile and run with Java 8+
# //
# // Notes:
# // - All implementations are optimized for clarity and performance.
# // - Brute-force methods are included for learning and comparison.
# // - Inline comments provide detailed step-by-step explanations.
# // - Some system design or advanced topics are conceptual or framework-based.
# //
# // License:
# // - Free to use for educational purposes, interview prep, and personal projects.
# //
# // ===========================================



# ===========================================
# Top 50 FAANG Interview Problems (Python)
# Part 1: Arrays & Strings (Problems 1–10)
# ===========================================

# -------------------------------------------
# Problem 1: Two Sum
# -------------------------------------------
"""
🧩 Problem Statement:
Given an array of integers nums and an integer target, return indices of the 
two numbers such that they add up to target.

💡 Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]  (because 2 + 7 = 9)

🔍 Concepts:
- Brute Force: Check all pairs (O(n^2))
- Optimized: Use HashMap to find complement in O(n)
"""

def two_sum_bruteforce(nums, target):
    """
    Brute-force: Check every pair
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []

def two_sum(nums, target):
    """
    Optimized Hash Map
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:
            return [seen[need], i]
        seen[x] = i
    return []

# -------------------------------------------
# Problem 2: Maximum Subarray (Kadane's)
# -------------------------------------------
"""
🧩 Problem Statement:
Find the contiguous subarray with the largest sum in an integer array.

💡 Example:
Input: [-2,1,-3,4,-1,2,1,-5,4]
Output: 6  (subarray [4,-1,2,1])

🔍 Concepts:
- Brute Force: Check all subarrays O(n^2)
- Optimized: Kadane’s algorithm O(n)
"""

def max_subarray_bruteforce(nums):
    best = float('-inf')
    for i in range(len(nums)):
        cur = 0
        for j in range(i, len(nums)):
            cur += nums[j]
            if cur > best:
                best = cur
    return best

def max_subarray(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

# -------------------------------------------
# Problem 3: Merge Intervals
# -------------------------------------------
"""
🧩 Problem Statement:
Given a list of intervals, merge all overlapping intervals.

💡 Example:
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]

🔍 Concepts:
- Sort intervals by start
- Merge overlapping by checking current interval with last merged
- O(n log n) due to sort
"""

def merge_intervals_bruteforce(intervals):
    intervals.sort()
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval[:])
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

# -------------------------------------------
# Problem 4: Container With Most Water
# -------------------------------------------
"""
🧩 Problem Statement:
Given an array of heights, find two lines that form a container containing the most water.

💡 Example:
Input: [1,8,6,2,5,4,8,3,7]
Output: 49

🔍 Concepts:
- Brute Force: Check all pairs O(n^2)
- Optimized: Two pointers O(n)
"""

def max_area_bruteforce(height):
    n = len(height)
    best = 0
    for i in range(n):
        for j in range(i+1, n):
            best = max(best, (j-i) * min(height[i], height[j]))
    return best

def max_area(height):
    l, r = 0, len(height)-1
    best = 0
    while l < r:
        best = max(best, (r-l) * min(height[l], height[r]))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return best

# -------------------------------------------
# Problem 5: 3Sum
# -------------------------------------------
"""
🧩 Problem Statement:
Find all unique triplets in the array which gives the sum of zero.

💡 Example:
Input: [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

🔍 Concepts:
- Sort array, fix one number, use two-pointer for remaining
- Avoid duplicates
- O(n^2)
"""

def three_sum(nums):
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n-2):
        if i>0 and nums[i]==nums[i-1]:
            continue
        l, r = i+1, n-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                l += 1; r -= 1
                while l<r and nums[l]==nums[l-1]: l += 1
                while l<r and nums[r]==nums[r+1]: r -= 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return res

# -------------------------------------------
# Problem 6: Longest Substring Without Repeating Characters
# -------------------------------------------
"""
🧩 Problem Statement:
Find the length of the longest substring without repeating characters.

💡 Example:
Input: "abcabcbb"
Output: 3  (substring "abc")

🔍 Concepts:
- Sliding window
- HashMap to store last index of character
- O(n)
"""

def length_of_longest_substring(s):
    last = {}
    start = 0
    best = 0
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best

# -------------------------------------------
# Problem 7: Trapping Rain Water
# -------------------------------------------
"""
🧩 Problem Statement:
Given n non-negative integers representing an elevation map, compute how much water it can trap.

💡 Example:
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

🔍 Concepts:
- Brute-force: Max to left & right O(n^2)
- Optimized: Two pointers O(n)
"""

def trap_bruteforce(height):
    n = len(height)
    water = 0
    for i in range(n):
        left = max(height[:i+1])
        right = max(height[i:])
        water += max(0, min(left, right) - height[i])
    return water

def trap(height):
    l, r = 0, len(height)-1
    left_max = right_max = water = 0
    while l < r:
        if height[l] < height[r]:
            if height[l] >= left_max:
                left_max = height[l]
            else:
                water += left_max - height[l]
            l += 1
        else:
            if height[r] >= right_max:
                right_max = height[r]
            else:
                water += right_max - height[r]
            r -= 1
    return water

# -------------------------------------------
# Problem 8: Product of Array Except Self
# -------------------------------------------
"""
🧩 Problem Statement:
Return an array output such that output[i] is product of all elements except nums[i] without division.

💡 Example:
Input: [1,2,3,4]
Output: [24,12,8,6]

🔍 Concepts:
- Prefix and suffix product arrays
- O(n) time, O(1) extra space (excluding output)
"""

def product_except_self(nums):
    n = len(nums)
    res = [1]*n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n-1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res

# -------------------------------------------
# Problem 9: Rotate Matrix
# -------------------------------------------
"""
🧩 Problem Statement:
Rotate n x n 2D matrix 90 degrees clockwise in-place.

💡 Example:
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

🔍 Concepts:
- Transpose matrix
- Reverse each row
"""

def rotate_matrix(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix

# -------------------------------------------
# Problem 10: Set Matrix Zeroes
# -------------------------------------------
"""
🧩 Problem Statement:
If an element in a matrix is 0, set its entire row and column to 0 in-place.

💡 Example:
Input: [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

🔍 Concepts:
- Use first row and column as markers
- O(1) extra space
"""

def set_zeroes(matrix):
    if not matrix: return matrix
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    for i in range(1,m):
        for j in range(1,n):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0
    for i in range(1,m):
        for j in range(1,n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if first_row_zero:
        for j in range(n): matrix[0][j] = 0
    if first_col_zero:
        for i in range(m): matrix[i][0] = 0
    return matrix











# ===========================================
# Top 50 FAANG Interview Problems (Python)
# Part 2: Linked Lists & Trees (Problems 11–20)
# ===========================================

# -------------------------------------------
# Problem 11: Reverse a Linked List
# -------------------------------------------
"""
🧩 Problem Statement:
Reverse a singly linked list (both iterative and recursive methods).

💡 Example:
Input: 1 -> 2 -> 3 -> 4 -> None
Output: 4 -> 3 -> 2 -> 1 -> None

🔍 Concepts:
- Iterative: Use prev, current, next pointers
- Recursive: Reverse rest and connect head at the end
"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list_iterative(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next      # store next
        cur.next = prev     # reverse link
        prev = cur          # move prev forward
        cur = nxt           # move current forward
    return prev

def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head

# -------------------------------------------
# Problem 12: Detect and Remove Loop in Linked List
# -------------------------------------------
"""
🧩 Problem Statement:
Detect a cycle in linked list and remove it.

💡 Example:
Input: 1->2->3->4->2 (cycle)
Output: 1->2->3->4->None

🔍 Concepts:
- Use Floyd’s cycle detection (slow & fast pointers)
- To remove: find start of cycle, set last node.next = None
"""

def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None

def remove_cycle(head):
    start = detect_cycle_start(head)
    if not start:
        return head
    cur = start
    while cur.next != start:
        cur = cur.next
    cur.next = None
    return head

# -------------------------------------------
# Problem 13: Merge Two Sorted Linked Lists
# -------------------------------------------
"""
🧩 Problem Statement:
Merge two sorted linked lists into one sorted list.

💡 Example:
Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4

🔍 Concepts:
- Use dummy head and iterate both lists
"""

def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    tail = dummy
    a, b = l1, l2
    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next

# -------------------------------------------
# Problem 14: LRU Cache Implementation
# -------------------------------------------
"""
🧩 Problem Statement:
Design LRU Cache with get and put in O(1) time.

🔍 Concepts:
- Doubly Linked List + HashMap
- Move accessed nodes to front
- Remove least recently used if capacity exceeded
"""

class DListNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}
        self.head = DListNode()
        self.tail = DListNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    def _add_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_front(node)
        return node.val

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = DListNode(key, value)
        self._add_front(node)
        self.cache[key] = node
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

# -------------------------------------------
# Problem 15: Lowest Common Ancestor (Binary Tree)
# -------------------------------------------
"""
🧩 Problem Statement:
Find LCA of two nodes in a binary tree.

💡 Example:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p=5, q=1
Output: 3

🔍 Concepts:
- DFS recursion
- If left & right return non-null, root is LCA
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right

# -------------------------------------------
# Problem 16: Serialize and Deserialize Binary Tree
# -------------------------------------------
"""
🧩 Problem Statement:
Serialize a binary tree to a string and deserialize it back to tree.

💡 Example:
Input: root = [1,2,3,null,null,4,5]
Output: same tree

🔍 Concepts:
- Use preorder traversal with sentinel for null
- Use iterator for deserialization
"""

def serialize(root):
    res = []
    def dfs(node):
        if not node:
            res.append('#')
            return
        res.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ','.join(res)

def deserialize(data):
    vals = iter(data.split(','))
    def dfs():
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.left = dfs()
        node.right = dfs()
        return node
    return dfs()

# -------------------------------------------
# Problem 17: Validate Binary Search Tree
# -------------------------------------------
"""
🧩 Problem Statement:
Check if a binary tree is a valid BST.

💡 Example:
Input: [2,1,3]
Output: True

🔍 Concepts:
- Recursively check left < node < right
- Use min/max bounds
"""

def is_valid_bst(root, low=float('-inf'), high=float('inf')):
    if not root:
        return True
    if not (low < root.val < high):
        return False
    return is_valid_bst(root.left, low, root.val) and is_valid_bst(root.right, root.val, high)

# -------------------------------------------
# Problem 18: Zigzag Level Order Traversal
# -------------------------------------------
"""
🧩 Problem Statement:
Return the zigzag level order of a binary tree (left->right, then right->left).

💡 Example:
Input: [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

🔍 Concepts:
- BFS level order
- Reverse every alternate level
"""

from collections import deque

def zigzag_level_order(root):
    if not root:
        return []
    res, q, left_to_right = [], deque([root]), True
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return res

# -------------------------------------------
# Problem 19: Diameter of Binary Tree
# -------------------------------------------
"""
🧩 Problem Statement:
Length of the longest path between any two nodes in a binary tree.

💡 Example:
Input: [1,2,3,4,5]
Output: 3

🔍 Concepts:
- DFS
- diameter = left height + right height
"""

def diameter_of_binary_tree(root):
    diameter = 0
    def height(node):
        nonlocal diameter
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        diameter = max(diameter, left + right)
        return 1 + max(left, right)
    height(root)
    return diameter

# -------------------------------------------
# Problem 20: Vertical Order Traversal
# -------------------------------------------
"""
🧩 Problem Statement:
Return the vertical order traversal of a binary tree.

💡 Example:
Input: [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]

🔍 Concepts:
- BFS with column index
- Sort by row then column
"""

from collections import defaultdict

def vertical_order_traversal(root):
    if not root:
        return []
    node_list = []
    q = deque([(root, 0, 0)])  # node, row, col
    while q:
        node, row, col = q.popleft()
        node_list.append((col, row, node.val))
        if node.left: q.append((node.left, row+1, col-1))
        if node.right: q.append((node.right, row+1, col+1))
    node_list.sort()
    res_dict = defaultdict(list)
    for col, row, val in node_list:
        res_dict[col].append(val)
    return [res_dict[x] for x in sorted(res_dict)]













# ===========================================
# Top 50 FAANG Interview Problems (Python)
# Part 3: Graphs & DP (Problems 21–30)
# ===========================================

# -------------------------------------------
# Problem 21: Clone Graph
# -------------------------------------------
"""
🧩 Problem Statement:
Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

💡 Example:
Input: Node with neighbors [[2,4],[1,3],[2,4],[1,3]]
Output: Deep copy graph

🔍 Concepts:
- DFS or BFS
- Use hashmap to track cloned nodes
"""

def clone_graph(node):
    if not node:
        return None
    old_to_new = {}
    def dfs(n):
        if n in old_to_new:
            return old_to_new[n]
        copy = Node(n.val)
        old_to_new[n] = copy
        for nei in n.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node)

# -------------------------------------------
# Problem 22: Number of Islands (DFS/BFS)
# -------------------------------------------
"""
🧩 Problem Statement:
Given a 2D grid map of '1's (land) and '0's (water), count the number of islands.

💡 Example:
Input: grid = [
  ["1","1","0","0"],
  ["1","1","0","0"],
  ["0","0","1","0"],
  ["0","0","0","1"]
]
Output: 3

🔍 Concepts:
- DFS or BFS to mark visited
- O(m*n)
"""

def num_islands(grid):
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    def dfs(i, j):
        if i<0 or i>=m or j<0 or j>=n or grid[i][j]=='0':
            return
        grid[i][j]='0'
        dfs(i+1,j); dfs(i-1,j); dfs(i,j+1); dfs(i,j-1)
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j]=='1':
                dfs(i,j)
                count += 1
    return count

# -------------------------------------------
# Problem 23: Word Ladder (BFS Shortest Path)
# -------------------------------------------
"""
🧩 Problem Statement:
Transform beginWord to endWord, changing one letter at a time with intermediate words in wordList. Return shortest path length.

💡 Example:
beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5 ("hit"->"hot"->"dot"->"dog"->"cog")

🔍 Concepts:
- BFS
- Each word is a node
- Change one character per step
"""

from collections import deque

def ladder_length(beginWord, endWord, wordList):
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    q = deque([(beginWord, 1)])
    while q:
        word, steps = q.popleft()
        if word == endWord:
            return steps
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i]+c+word[i+1:]
                if next_word in wordSet:
                    wordSet.remove(next_word)
                    q.append((next_word, steps+1))
    return 0

# -------------------------------------------
# Problem 24: Topological Sort
# -------------------------------------------
"""
🧩 Problem Statement:
Return a topological ordering of a directed acyclic graph (DAG).

💡 Example:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]

🔍 Concepts:
- Kahn's algorithm (BFS)
- Maintain in-degree
"""

def topological_sort(numCourses, prerequisites):
    from collections import deque, defaultdict
    graph = defaultdict(list)
    in_deg = [0]*numCourses
    for dest, src in prerequisites:
        graph[src].append(dest)
        in_deg[dest] += 1
    q = deque([i for i in range(numCourses) if in_deg[i]==0])
    res = []
    while q:
        node = q.popleft()
        res.append(node)
        for nei in graph[node]:
            in_deg[nei]-=1
            if in_deg[nei]==0:
                q.append(nei)
    return res if len(res)==numCourses else []

# -------------------------------------------
# Problem 25: Detect Cycle in Directed Graph
# -------------------------------------------
"""
🧩 Problem Statement:
Detect if a directed graph has a cycle.

💡 Example:
Input: [[1,2],[2,3],[3,1]]
Output: True

🔍 Concepts:
- DFS with recursion stack
"""

def has_cycle(graph):
    visited, rec_stack = set(), set()
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for nei in graph[node]:
            if nei not in visited and dfs(nei):
                return True
            elif nei in rec_stack:
                return True
        rec_stack.remove(node)
        return False
    for node in graph:
        if node not in visited and dfs(node):
            return True
    return False

# -------------------------------------------
# Problem 26: Dijkstra's Shortest Path
# -------------------------------------------
"""
🧩 Problem Statement:
Find shortest paths from a source to all vertices in a weighted graph.

🔍 Concepts:
- Priority queue (min-heap)
- O(E log V)
"""

import heapq

def dijkstra(graph, src):
    """
    graph: adjacency list {node: [(neighbor, weight), ...]}
    src: starting node
    """
    dist = {node: float('inf') for node in graph}
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for nei, w in graph[node]:
            if dist[node] + w < dist[nei]:
                dist[nei] = dist[node] + w
                heapq.heappush(heap, (dist[nei], nei))
    return dist

# -------------------------------------------
# Problem 27: Longest Increasing Subsequence
# -------------------------------------------
"""
🧩 Problem Statement:
Find length of longest increasing subsequence in an array.

💡 Example:
Input: [10,9,2,5,3,7,101,18]
Output: 4 (subsequence: [2,3,7,101])

🔍 Concepts:
- DP: dp[i] = max dp[j]+1 for j<i if nums[j]<nums[i]
- O(n^2)
- Optimized O(n log n) with binary search
"""

def length_of_lis(nums):
    dp = [1]*len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp) if dp else 0

# -------------------------------------------
# Problem 28: 0/1 Knapsack Problem
# -------------------------------------------
"""
🧩 Problem Statement:
Given weights and values, maximize value without exceeding weight limit.

💡 Example:
weights=[1,3,4], values=[15,20,30], W=4
Output: 35 (choose items 1 and 3)

🔍 Concepts:
- DP: dp[i][w] = max value using first i items with capacity w
- Time/Space: O(n*W)
"""

def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for w in range(W+1):
            if weights[i-1]<=w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]]+values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]

# -------------------------------------------
# Problem 29: Coin Change (Minimum Coins)
# -------------------------------------------
"""
🧩 Problem Statement:
Given coins and amount, return minimum number of coins to make amount.

💡 Example:
coins=[1,2,5], amount=11
Output: 3 (5+5+1)

🔍 Concepts:
- DP: dp[i] = min(dp[i-coin]+1 for coin in coins)
- Initialize dp[0]=0, others=inf
"""

def coin_change(coins, amount):
    dp = [float('inf')]*(amount+1)
    dp[0] = 0
    for a in range(1, amount+1):
        for c in coins:
            if a>=c:
                dp[a] = min(dp[a], dp[a-c]+1)
    return dp[amount] if dp[amount]!=float('inf') else -1

# -------------------------------------------
# Problem 30: Edit Distance (Levenshtein Distance)
# -------------------------------------------
"""
🧩 Problem Statement:
Given two strings, find minimum number of operations to convert word1 to word2
(insert, delete, replace).

💡 Example:
word1="horse", word2="ros"
Output: 3

🔍 Concepts:
- DP: dp[i][j] = min operations to convert first i chars to first j chars
"""

def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            if word1[i-1]==word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]











# ===========================================
# Top 50 FAANG Interview Problems (Python)
# Part 4: DP Second Half + Heaps, Stacks & Queues (Problems 31–40)
# ===========================================

# -------------------------------------------
# Problem 31: Longest Palindromic Substring
# -------------------------------------------
"""
🧩 Problem Statement:
Given a string s, return the longest palindromic substring.

💡 Example:
Input: "babad"
Output: "bab" (or "aba")

🔍 Concepts:
- Expand around center O(n^2)
- Dynamic programming O(n^2)
"""

def longest_palindrome(s):
    if not s: return ""
    start, end = 0, 0
    for i in range(len(s)):
        # Odd length palindrome
        l, r = i, i
        while l>=0 and r<len(s) and s[l]==s[r]:
            if r-l > end-start:
                start, end = l, r
            l -= 1; r += 1
        # Even length palindrome
        l, r = i, i+1
        while l>=0 and r<len(s) and s[l]==s[r]:
            if r-l > end-start:
                start, end = l, r
            l -= 1; r += 1
    return s[start:end+1]

# -------------------------------------------
# Problem 32: Partition Equal Subset Sum
# -------------------------------------------
"""
🧩 Problem Statement:
Given a list of positive integers, check if it can be partitioned into two subsets with equal sum.

💡 Example:
Input: [1,5,11,5]
Output: True

🔍 Concepts:
- Reduce to subset sum problem
- DP boolean array for achievable sums
"""

def can_partition(nums):
    total = sum(nums)
    if total%2 !=0: return False
    target = total//2
    dp = [False]*(target+1)
    dp[0] = True
    for num in nums:
        for t in range(target, num-1, -1):
            dp[t] = dp[t] or dp[t-num]
    return dp[target]

# -------------------------------------------
# Problem 33: Maximum Product Subarray
# -------------------------------------------
"""
🧩 Problem Statement:
Find contiguous subarray with maximum product.

💡 Example:
Input: [2,3,-2,4]
Output: 6

🔍 Concepts:
- Keep track of max and min product at each position
- Because negative * negative = positive
"""

def max_product(nums):
    if not nums: return 0
    cur_max = cur_min = res = nums[0]
    for n in nums[1:]:
        tmp = cur_max
        cur_max = max(n, n*cur_max, n*cur_min)
        cur_min = min(n, n*tmp, n*cur_min)
        res = max(res, cur_max)
    return res

# -------------------------------------------
# Problem 34: House Robber (I)
# -------------------------------------------
"""
🧩 Problem Statement:
Max sum of non-adjacent houses.

💡 Example:
Input: [2,7,9,3,1]
Output: 12

🔍 Concepts:
- DP: dp[i] = max(dp[i-1], dp[i-2]+nums[i])
"""

def rob(nums):
    prev = curr = 0
    for n in nums:
        prev, curr = curr, max(curr, prev+n)
    return curr

# -------------------------------------------
# Problem 35: House Robber II (Circular Houses)
# -------------------------------------------
"""
🧩 Problem Statement:
Max sum of non-adjacent houses in a circle.

💡 Example:
Input: [2,3,2]
Output: 3

🔍 Concepts:
- Rob either exclude first or exclude last
"""

def rob_circle(nums):
    if len(nums)==1: return nums[0]
    def rob_linear(houses):
        prev = curr = 0
        for n in houses:
            prev, curr = curr, max(curr, prev+n)
        return curr
    return max(rob_linear(nums[1:]), rob_linear(nums[:-1]))

# -------------------------------------------
# Problem 36: Unique Paths (with Obstacles)
# -------------------------------------------
"""
🧩 Problem Statement:
Find unique paths from top-left to bottom-right in a grid with obstacles.

💡 Example:
Input: obstacleGrid=[[0,0,0],[0,1,0],[0,0,0]]
Output: 2

🔍 Concepts:
- DP: dp[i][j] = dp[i-1][j] + dp[i][j-1]
- Handle obstacle cells
"""

def unique_paths_with_obstacles(obstacleGrid):
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if obstacleGrid[i][j]==1:
                dp[i][j]=0
            elif i==0 and j==0:
                dp[i][j]=1
            else:
                dp[i][j]= (dp[i-1][j] if i>0 else 0) + (dp[i][j-1] if j>0 else 0)
    return dp[m-1][n-1]

# -------------------------------------------
# Problem 37: Palindrome Partitioning (Min Cuts)
# -------------------------------------------
"""
🧩 Problem Statement:
Partition string into palindromes with minimum cuts.

💡 Example:
Input: "aab"
Output: 1 ("aa|b")

🔍 Concepts:
- DP with palindrome table
- dp[i] = min cuts for s[:i+1]
"""

def min_cut_palindrome(s):
    n = len(s)
    dp = [0]*n
    pal = [[False]*n for _ in range(n)]
    for i in range(n):
        min_cut = i
        for j in range(i+1):
            if s[j]==s[i] and (i-j<2 or pal[j+1][i-1]):
                pal[j][i]=True
                min_cut = 0 if j==0 else min(min_cut, dp[j-1]+1)
        dp[i]=min_cut
    return dp[-1]

# -------------------------------------------
# Problem 38: Merge K Sorted Lists
# -------------------------------------------
"""
🧩 Problem Statement:
Merge k sorted linked lists into one sorted list.

💡 Concepts:
- Min-heap O(N log k)
"""

import heapq

def merge_k_lists(lists):
    heap = []
    for l in lists:
        while l:
            heapq.heappush(heap, l.val)
            l = l.next
    dummy = ListNode(0)
    cur = dummy
    while heap:
        cur.next = ListNode(heapq.heappop(heap))
        cur = cur.next
    return dummy.next

# -------------------------------------------
# Problem 39: Sliding Window Maximum
# -------------------------------------------
"""
🧩 Problem Statement:
Find maximum in every sliding window of size k.

💡 Example:
Input: nums=[1,3,-1,-3,5,3,6,7], k=3
Output: [3,3,5,5,6,7]

🔍 Concepts:
- Deque to store indexes of useful elements
"""

from collections import deque

def max_sliding_window(nums, k):
    dq = deque()
    res = []
    for i, n in enumerate(nums):
        while dq and dq[0]<=i-k:
            dq.popleft()
        while dq and nums[dq[-1]]<n:
            dq.pop()
        dq.append(i)
        if i>=k-1:
            res.append(nums[dq[0]])
    return res

# -------------------------------------------
# Problem 40: Min Stack Implementation
# -------------------------------------------
"""
🧩 Problem Statement:
Stack supporting push, pop, top, getMin in O(1).

🔍 Concepts:
- Use auxiliary stack to track minimums
"""

class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val<=self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]











# ===========================================
# Top 50 FAANG Interview Problems (Python)
# Part 5: Backtracking + System Design / Advanced Python (Problems 41–50)
# ===========================================

# -------------------------------------------
# Problem 41: N-Queens Problem
# -------------------------------------------
"""
🧩 Problem Statement:
Place N queens on an N×N chessboard so that no two queens attack each other.
Return all distinct solutions.

💡 Example:
Input: 4
Output: [[".Q..","...Q","Q...","..Q."], ...]

🔍 Concepts:
- Backtracking
- Track columns and diagonals
"""

def solve_n_queens(n):
    res = []
    def backtrack(row, cols, diag1, diag2, path):
        if row==n:
            board = ["."*i+"Q"+"."*(n-i-1) for i in path]
            res.append(board)
            return
        for col in range(n):
            if col in cols or (row+col) in diag1 or (row-col) in diag2:
                continue
            backtrack(row, cols|{col}, diag1|{row+col}, diag2|{row-col}, path+[col])
            backtrack(row+1, cols|{col}, diag1|{row+col}, diag2|{row-col}, path+[col])
    backtrack(0,set(),set(),set(),[])
    return res

# -------------------------------------------
# Problem 42: Sudoku Solver
# -------------------------------------------
"""
🧩 Problem Statement:
Solve a 9x9 Sudoku puzzle in-place.

💡 Concepts:
- Backtracking
- Check row, column, and 3x3 box validity
"""

def solve_sudoku(board):
    def is_valid(r, c, ch):
        for i in range(9):
            if board[r][i]==ch or board[i][c]==ch or board[3*(r//3)+i//3][3*(c//3)+i%3]==ch:
                return False
        return True
    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j]=='.':
                    for ch in '123456789':
                        if is_valid(i,j,ch):
                            board[i][j]=ch
                            if backtrack(): return True
                            board[i][j]='.'
                    return False
        return True
    backtrack()

# -------------------------------------------
# Problem 43: Subsets / Permutations Generation
# -------------------------------------------
"""
🧩 Problem Statement:
Generate all subsets or permutations of a list.

💡 Example:
Input: [1,2,3]
Subsets: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
Permutations: [[1,2,3],[1,3,2],...]
"""

def subsets(nums):
    res = []
    def backtrack(path, i):
        if i==len(nums):
            res.append(path[:])
            return
        backtrack(path, i+1)
        backtrack(path+[nums[i]], i+1)
    backtrack([],0)
    return res

def permutations(nums):
    res = []
    def backtrack(path, used):
        if len(path)==len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if not used[i]:
                used[i]=True
                path.append(nums[i])
                backtrack(path, used)
                path.pop()
                used[i]=False
    backtrack([], [False]*len(nums))
    return res

# -------------------------------------------
# Problem 44: Design a URL Shortener (like TinyURL)
# -------------------------------------------
"""
🧩 Problem Statement:
Create a system to encode a URL to a short URL and decode back.

💡 Concepts:
- Hash map or counter-based encoding
"""

class Codec:
    def __init__(self):
        self.url_map = {}
        self.counter = 1
    def encode(self, longUrl):
        shortUrl = "http://tinyurl.com/"+str(self.counter)
        self.url_map[shortUrl]=longUrl
        self.counter+=1
        return shortUrl
    def decode(self, shortUrl):
        return self.url_map.get(shortUrl,"")

# -------------------------------------------
# Problem 45: Design a Rate Limiter
# -------------------------------------------
"""
🧩 Problem Statement:
Limit number of requests per user per time window.

💡 Concepts:
- Use timestamp queue per user
- Check if requests exceed limit
"""

from collections import deque
import time

class RateLimiter:
    def __init__(self, limit, interval):
        self.limit = limit
        self.interval = interval
        self.user_map = {}
    def allow(self, user):
        now = time.time()
        if user not in self.user_map:
            self.user_map[user] = deque()
        q = self.user_map[user]
        while q and q[0]<=now-self.interval:
            q.popleft()
        if len(q)<self.limit:
            q.append(now)
            return True
        return False

# -------------------------------------------
# Problem 46: Design a Distributed Cache (LRU)
# -------------------------------------------
"""
🧩 Problem Statement:
LRU cache with get, put operations.

💡 Concepts:
- Doubly Linked List + Hashmap
"""

class DListNode:
    def __init__(self,key=0,val=0):
        self.key=key; self.val=val; self.prev=None; self.next=None

class LRUCache:
    def __init__(self, capacity):
        self.cap=capacity
        self.cache={}
        self.head=DListNode()
        self.tail=DListNode()
        self.head.next=self.tail
        self.tail.prev=self.head
    def _remove(self,node):
        node.prev.next=node.next
        node.next.prev=node.prev
    def _add_front(self,node):
        node.next=self.head.next
        node.prev=self.head
        self.head.next.prev=node
        self.head.next=node
    def get(self,key):
        if key not in self.cache: return -1
        node=self.cache[key]
        self._remove(node)
        self._add_front(node)
        return node.val
    def put(self,key,val):
        if key in self.cache:
            node=self.cache[key]
            node.val=val
            self._remove(node)
            self._add_front(node)
        else:
            if len(self.cache)==self.cap:
                lru=self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            node=DListNode(key,val)
            self.cache[key]=node
            self._add_front(node)

# -------------------------------------------
# Problem 47: Design Twitter Feed System
# -------------------------------------------
"""
🧩 Problem Statement:
Implement a simplified Twitter with postTweet, follow, unfollow, getNewsFeed

💡 Concepts:
- Maintain tweet list per user
- Heap for recent tweets
"""

import heapq

class Twitter:
    def __init__(self):
        self.tweets = {} # user: [(time,id)]
        self.following = {} # user:set
        self.time = 0
    def postTweet(self,userId,tweetId):
        self.tweets.setdefault(userId,[]).append((self.time,tweetId))
        self.time+=1
    def getNewsFeed(self,userId):
        res=[]
        heap=[]
        users=self.following.get(userId,set()) | {userId}
        for u in users:
            for t in self.tweets.get(u,[]):
                heapq.heappush(heap,(-t[0],t[1]))
        for _ in range(10):
            if heap:
                res.append(heapq.heappop(heap)[1])
        return res
    def follow(self,followerId,followeeId):
        self.following.setdefault(followerId,set()).add(followeeId)
    def unfollow(self,followerId,followeeId):
        self.following.setdefault(followerId,set()).discard(followeeId)

# -------------------------------------------
# Problem 48: Implement your own @decorator
# -------------------------------------------
"""
🧩 Problem Statement:
Python decorator to log function call

💡 Concepts:
- @decorator syntax
"""

def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        res = func(*args, **kwargs)
        print(f"{func.__name__} returned {res}")
        return res
    return wrapper

# -------------------------------------------
# Problem 49: Custom Context Manager
# -------------------------------------------
"""
🧩 Problem Statement:
Use __enter__ and __exit__ to handle resources
"""

class MyContext:
    def __enter__(self):
        print("Entering context")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context")

# -------------------------------------------
# Problem 50: Threading vs Multiprocessing / GIL & Memory Optimization
# -------------------------------------------
"""
🧩 Problem Statement:
Understand performance differences, GIL effect, and memory optimization techniques

💡 Concepts:
- Use multiprocessing for CPU bound
- Use threading for IO bound
- GIL prevents multiple native threads executing Python bytecode simultaneously
- Memory profiling with memory_profiler
"""

import threading
import multiprocessing

def cpu_task(n):
    s=0
    for i in range(n):
        s+=i*i
    return s

def io_task():
    import time
    time.sleep(1)

def run_threads():
    threads = [threading.Thread(target=io_task) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

def run_processes():
    procs = [multiprocessing.Process(target=cpu_task, args=(10**6,)) for _ in range(5)]
    for p in procs: p.start()
    for p in procs: p.join()













# ===========================================
# Bonus: Advanced Python Internals
# ===========================================

# -------------------------------------------
# A) Implement your own @decorator
# -------------------------------------------
"""
🧩 Problem Statement:
Create a decorator that logs function calls and results.

💡 Concepts:
- Decorators wrap a function
- Can access args, kwargs, and return value
"""

def log_decorator(func):
    """
    This decorator prints the function name, arguments, and result.
    """
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

# Example usage
@log_decorator
def add(a, b):
    return a + b

# add(3,4)  -> logs calling and returned 7

# -------------------------------------------
# B) Custom Context Manager using __enter__ and __exit__
# -------------------------------------------
"""
🧩 Problem Statement:
Implement a context manager to handle resources safely.

💡 Concepts:
- __enter__ called at start of 'with' block
- __exit__ called at end (even on exceptions)
"""

class MyContext:
    def __enter__(self):
        print("Entering context")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Exception caught: {exc_type}, {exc_val}")
        print("Exiting context")
        # return True to suppress exception if needed
        return False

# Example usage
# with MyContext() as ctx:
#     print("Inside context")

# -------------------------------------------
# C) Threading vs Multiprocessing performance
# -------------------------------------------
"""
🧩 Problem Statement:
Compare performance of threading vs multiprocessing for CPU-bound and IO-bound tasks.

💡 Concepts:
- Threading: good for IO-bound (network, disk)
- Multiprocessing: good for CPU-bound (heavy computations)
- Python threads limited by GIL for CPU tasks
"""

import threading
import multiprocessing
import time

def cpu_task(n):
    s=0
    for i in range(n):
        s+=i*i
    return s

def io_task():
    time.sleep(1)

def run_threads():
    threads = [threading.Thread(target=io_task) for _ in range(5)]
    start = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    print("Threading IO-bound time:", time.time()-start)

def run_processes():
    procs = [multiprocessing.Process(target=cpu_task, args=(10**6,)) for _ in range(5)]
    start = time.time()
    for p in procs: p.start()
    for p in procs: p.join()
    print("Multiprocessing CPU-bound time:", time.time()-start)

# -------------------------------------------
# D) GIL (Global Interpreter Lock) explained
# -------------------------------------------
"""
🧩 Problem Statement:
Understand why Python threads cannot execute bytecode in parallel for CPU-bound tasks.

💡 Concepts:
- GIL is a mutex that protects Python objects from concurrent access
- Prevents multiple threads from running Python bytecode simultaneously
- Threads can still run in parallel for IO-bound tasks
- Use multiprocessing to bypass GIL for CPU tasks
"""

# Example: CPU-bound task with threading will not speed up due to GIL
# Use multiprocessing to fully utilize multiple cores

# -------------------------------------------
# E) Memory profiling and optimization techniques
# -------------------------------------------
"""
🧩 Problem Statement:
Track memory usage and optimize Python programs.

💡 Concepts:
- memory_profiler: @profile decorator to check memory usage
- Use generators instead of lists for large datasets
- Use __slots__ in classes to reduce memory
- Avoid unnecessary copies
"""

# Example using generator
def large_numbers_gen(n):
    for i in range(n):
        yield i*i  # yields one at a time, saves memory

# Example using __slots__
class Point:
    __slots__ = ['x','y']  # prevents __dict__, saves memory
    def __init__(self,x,y):
        self.x=x
        self.y=y

# memory_profiler usage (run with 'mprof' or '@profile' decorator externally)
# @profile
# def my_function():
#     a = [i*i for i in range(1000000)]
#     return sum(a)
