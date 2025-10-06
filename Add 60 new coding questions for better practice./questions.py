# ========================
# DSA Practice Questions - Python
# Author: Mohit Kourav
# Description: 60 commonly asked DSA questions with definitions, code, explanation, and complexity
# ========================

# ========================
# Arrays & Strings (10)
# ========================

"""
This section contains 10 common problems on arrays and strings. 
These problems help practice contiguous data storage, 
array/string manipulation, searching, sorting, sliding window, 
and two-pointer techniques. Each problem improves problem-solving 
skills for coding interviews.
"""

# 1. Two Sum
# Definition: Given an array of integers, find two numbers that add up to a specific target.
# Time Complexity: O(n) | Space Complexity: O(n)
def two_sum(nums, target):
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i
    return []

# Explanation:
# - Store numbers and their indices in a hash map.
# - For each number, check if (target - num) exists in the map.
# - If yes, return indices; otherwise, add num to map.

# Example:
print(two_sum([2,7,11,15], 9))  # Output: [0, 1]


# 2. Maximum Subarray (Kadane’s Algorithm)
# Definition: Find the contiguous subarray with the largest sum.
# Time Complexity: O(n) | Space Complexity: O(1)
def max_subarray(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

# Explanation:
# - Iterate through array, for each element, decide to include in current sum or start new subarray.
# - Track the maximum sum encountered.

# Example:
print(max_subarray([-2,1,-3,4,-1,2,1,-5,4]))  # Output: 6


# 3. Rotate Array
# Definition: Rotate array to the right by k steps.
# Time Complexity: O(n) | Space Complexity: O(1)
def rotate_array(nums, k):
    n = len(nums)
    k %= n
    nums[:] = nums[-k:] + nums[:-k]
    return nums

# Example:
print(rotate_array([1,2,3,4,5,6,7], 3))  # Output: [5,6,7,1,2,3,4]


# 4. Product of Array Except Self
# Definition: Return an array such that output[i] is product of all elements except nums[i].
# Time Complexity: O(n) | Space Complexity: O(1) (ignoring output array)
def product_except_self(nums):
    n = len(nums)
    output = [1]*n
    left = 1
    for i in range(n):
        output[i] = left
        left *= nums[i]
    right = 1
    for i in range(n-1, -1, -1):
        output[i] *= right
        right *= nums[i]
    return output

# Example:
print(product_except_self([1,2,3,4]))  # Output: [24,12,8,6]


# 5. Trapping Rain Water
# Definition: Compute how much water can be trapped after raining.
# Time Complexity: O(n) | Space Complexity: O(n)
def trap(height):
    if not height:
        return 0
    n = len(height)
    left_max = [0]*n
    right_max = [0]*n
    left_max[0] = height[0]
    for i in range(1,n):
        left_max[i] = max(left_max[i-1], height[i])
    right_max[-1] = height[-1]
    for i in range(n-2,-1,-1):
        right_max[i] = max(right_max[i+1], height[i])
    water = 0
    for i in range(n):
        water += min(left_max[i], right_max[i]) - height[i]
    return water

# Example:
print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))  # Output: 6


# 6. Valid Palindrome
# Definition: Check if a string is a palindrome ignoring non-alphanumeric and case.
# Time Complexity: O(n) | Space Complexity: O(1)
def is_palindrome(s):
    left, right = 0, len(s)-1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

# Example:
print(is_palindrome("A man, a plan, a canal: Panama"))  # Output: True


# 7. Longest Substring Without Repeating Characters
# Definition: Find length of longest substring without repeating characters.
# Time Complexity: O(n) | Space Complexity: O(min(n, a)) where a is alphabet size
def length_of_longest_substring(s):
    char_set = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right-left+1)
    return max_len

# Example:
print(length_of_longest_substring("abcabcbb"))  # Output: 3


# 8. Merge Intervals
# Definition: Merge all overlapping intervals.
# Time Complexity: O(n log n) | Space Complexity: O(n)
def merge_intervals(intervals):
    intervals.sort(key=lambda x:x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

# Example:
print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))  # Output: [[1,6],[8,10],[15,18]]


# 9. Group Anagrams
# Definition: Group words that are anagrams of each other.
# Time Complexity: O(n*k log k) | Space Complexity: O(n*k), n = #words, k = avg length
from collections import defaultdict
def group_anagrams(strs):
    anagrams = defaultdict(list)
    for word in strs:
        anagrams[tuple(sorted(word))].append(word)
    return list(anagrams.values())

# Example:
print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# Output: [['eat','tea','ate'], ['tan','nat'], ['bat']]


# 10. Move Zeroes
# Definition: Move all zeroes to the end of the array while maintaining order of other elements.
# Time Complexity: O(n) | Space Complexity: O(1)
def move_zeroes(nums):
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[i], nums[j] = nums[j], nums[i]
            j += 1
    return nums

# Example:
print(move_zeroes([0,1,0,3,12]))  # Output: [1,3,12,0,0]


# ========================
# Linked List (8)
# ========================

"""'
This section contains 8 problems on singly and doubly linked lists. 
Focus is on insertion, deletion, reversing, detecting cycles, 
and pointer manipulation. These problems help strengthen 
understanding of dynamic data structures.
"""

# 1. Reverse a Linked List
# Definition: Reverse a singly linked list.
# Time Complexity: O(n) | Space Complexity: O(1)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

# Explanation:
# - Use three pointers: prev, curr, next
# - Iteratively reverse links
# - Return new head

# Example:
# Input: 1->2->3->None, Output: 3->2->1->None

# 2. Detect Cycle in Linked List
# Definition: Detect if a linked list has a cycle using Floyd’s algorithm.
# Time Complexity: O(n) | Space Complexity: O(1)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# 3. Merge Two Sorted Linked Lists
# Definition: Merge two sorted linked lists into one sorted list.
# Time Complexity: O(n+m) | Space Complexity: O(1)
def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    tail = dummy
    while l1 and l2:
        if l1.val < l2.val:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next

# 4. Remove N-th Node From End
# Definition: Remove the n-th node from the end of the list.
# Time Complexity: O(n) | Space Complexity: O(1)
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    first = second = dummy
    for _ in range(n+1):
        first = first.next
    while first:
        first = first.next
        second = second.next
    second.next = second.next.next
    return dummy.next

# 5. Copy List with Random Pointer
# Definition: Deep copy a linked list where each node has a random pointer.
# Time Complexity: O(n) | Space Complexity: O(1)
def copy_random_list(head):
    if not head:
        return None
    # Step1: Duplicate nodes
    curr = head
    while curr:
        nxt = curr.next
        copy = ListNode(curr.val)
        curr.next = copy
        copy.next = nxt
        curr = nxt
    # Step2: Set random pointers
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    # Step3: Separate lists
    dummy = ListNode(0)
    copy_curr, curr = dummy, head
    while curr:
        copy_curr.next = curr.next
        curr.next = curr.next.next
        curr = curr.next
        copy_curr = copy_curr.next
    return dummy.next

# 6. Palindrome Linked List
# Definition: Check if a linked list is a palindrome.
# Time Complexity: O(n) | Space Complexity: O(1)
def is_palindrome_linked_list(head):
    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # Reverse second half
    prev = None
    curr = slow
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    # Compare halves
    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True

# 7. Flatten Multilevel Linked List
# Definition: Flatten a linked list where nodes may have child pointers to another list.
# Time Complexity: O(n) | Space Complexity: O(1)
def flatten(head):
    curr = head
    stack = []
    while curr:
        if curr.child:
            if curr.next:
                stack.append(curr.next)
            curr.next = curr.child
            curr.child = None
        if not curr.next and stack:
            curr.next = stack.pop()
        curr = curr.next
    return head

# 8. Intersection of Two Linked Lists
# Definition: Find intersection node of two singly linked lists.
# Time Complexity: O(n+m) | Space Complexity: O(1)
def get_intersection_node(headA, headB):
    a, b = headA, headB
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a


# ========================
# Stack & Queue (5)
# ========================

"""
This section contains 5 problems on stacks and queues. 
It includes classic stack/queue operations, monotonic stacks, 
deque, and circular queue problems. These exercises improve 
sequential processing, backtracking, and order-based logic skills.
"""

# 1. Valid Parentheses
# Definition: Check if a string containing (), {}, [] is valid.
# Time Complexity: O(n) | Space Complexity: O(n)
def is_valid_parentheses(s):
    stack = []
    mapping = {')':'(', '}':'{', ']':'['}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return not stack

# Example:
print(is_valid_parentheses("()[]{}"))  # Output: True

# 2. Min Stack
# Definition: Implement a stack that returns the minimum element in O(1).
# Time Complexity: O(1) per operation | Space Complexity: O(n)
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]

# 3. Implement Queue using Stacks
# Definition: Implement a queue using two stacks.
# Time Complexity: O(1) amortized | Space Complexity: O(n)
class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self.peek()
        return self.out_stack.pop()

    def peek(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack

# 4. Sliding Window Maximum
# Definition: Find maximum in each sliding window of size k.
# Time Complexity: O(n) | Space Complexity: O(k)
from collections import deque
def max_sliding_window(nums, k):
    dq = deque()
    res = []
    for i, num in enumerate(nums):
        while dq and dq[0] <= i-k:
            dq.popleft()
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)
        if i >= k-1:
            res.append(nums[dq[0]])
    return res

# Example:
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))  # Output: [3,3,5,5,6,7]

# 5. Next Greater Element
# Definition: Find the next greater element for each element in array.
# Time Complexity: O(n) | Space Complexity: O(n)
def next_greater_elements(nums):
    res = [-1]*len(nums)
    stack = []
    for i in range(len(nums)-1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            res[i] = stack[-1]
        stack.append(nums[i])
    return res

# Example:
print(next_greater_elements([4,5,2,10]))  # Output: [5,10,10,-1]

# ========================
# Trees & Graphs (12)
# ========================

"""
This section contains 12 problems on trees and graphs. 
Covers binary trees, BSTs, N-ary trees, graph traversal (BFS/DFS), 
shortest paths, minimum spanning trees, and cycle detection. 
Helps develop understanding of hierarchical and relational data structures 
and traversal algorithms.
"""

# 1. Maximum Depth of Binary Tree
# Definition: Find the maximum depth of a binary tree.
# Time Complexity: O(n) | Space Complexity: O(h)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# 2. Lowest Common Ancestor (Binary Tree)
# Definition: Find the lowest common ancestor of two nodes in a binary tree.
# Time Complexity: O(n) | Space Complexity: O(h)
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    return root if left and right else left or right

# 3. Diameter of Binary Tree
# Definition: Length of the longest path between any two nodes.
# Time Complexity: O(n) | Space Complexity: O(h)
def diameter_of_binary_tree(root):
    diameter = 0
    def dfs(node):
        nonlocal diameter
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        diameter = max(diameter, left + right)
        return 1 + max(left, right)
    dfs(root)
    return diameter

# 4. Serialize and Deserialize Binary Tree
# Time Complexity: O(n) | Space Complexity: O(n)
def serialize(root):
    res = []
    def helper(node):
        if node is None:
            res.append('X')
            return
        res.append(str(node.val))
        helper(node.left)
        helper(node.right)
    helper(root)
    return ','.join(res)

def deserialize(data):
    vals = iter(data.split(','))
    def helper():
        val = next(vals)
        if val == 'X':
            return None
        node = TreeNode(int(val))
        node.left = helper()
        node.right = helper()
        return node
    return helper()

# 5. Binary Tree Level Order Traversal
# Time Complexity: O(n) | Space Complexity: O(n)
from collections import deque
def level_order(root):
    if not root:
        return []
    res = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level)
    return res

# 6. Number of Islands (DFS)
# Definition: Count number of islands in 2D grid.
# Time Complexity: O(M*N) | Space Complexity: O(M*N)
def num_islands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    def dfs(r,c):
        if r<0 or r>=rows or c<0 or c>=cols or grid[r][c]=='0':
            return
        grid[r][c] = '0'
        dfs(r+1,c); dfs(r-1,c); dfs(r,c+1); dfs(r,c-1)
    count=0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]=='1':
                dfs(r,c)
                count+=1
    return count

# 7. Clone Graph (DFS)
# Definition: Deep copy of a graph.
# Time Complexity: O(V+E) | Space Complexity: O(V)
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []

def clone_graph(node):
    old_to_new = {}
    def dfs(n):
        if n in old_to_new:
            return old_to_new[n]
        copy = Node(n.val)
        old_to_new[n] = copy
        for nei in n.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node) if node else None

# 8. Course Schedule (Topological Sort)
# Time Complexity: O(V+E) | Space Complexity: O(V+E)
def can_finish(num_courses, prerequisites):
    from collections import defaultdict, deque
    graph = defaultdict(list)
    indegree = [0]*num_courses
    for u,v in prerequisites:
        graph[v].append(u)
        indegree[u] += 1
    queue = deque([i for i in range(num_courses) if indegree[i]==0])
    visited = 0
    while queue:
        node = queue.popleft()
        visited +=1
        for nei in graph[node]:
            indegree[nei]-=1
            if indegree[nei]==0:
                queue.append(nei)
    return visited==num_courses

# 9. Dijkstra’s Shortest Path
# Time Complexity: O(E log V) | Space Complexity: O(V+E)
import heapq
def dijkstra(graph, start):
    pq = [(0, start)]
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    while pq:
        d, node = heapq.heappop(pq)
        if d>dist[node]:
            continue
        for nei, w in graph[node]:
            if dist[node]+w<dist[nei]:
                dist[nei]=dist[node]+w
                heapq.heappush(pq, (dist[nei], nei))
    return dist

# 10. Kruskal’s MST
# Time Complexity: O(E log E) | Space Complexity: O(V)
class UnionFind:
    def __init__(self,n):
        self.parent = list(range(n))
    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        px,py=self.find(x),self.find(y)
        if px==py: return False
        self.parent[px]=py
        return True

def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x:x[2])
    mst=[]
    for u,v,w in edges:
        if uf.union(u,v):
            mst.append((u,v,w))
    return mst

# 11. Prim’s MST
# Time Complexity: O(E log V) | Space Complexity: O(V+E)
def prim_mst(graph):
    import heapq
    start = next(iter(graph))
    visited = set([start])
    edges = [(w, start, v) for v, w in graph[start]]
    heapq.heapify(edges)
    mst = []
    while edges:
        w, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u,v,w))
            for to, weight in graph[v]:
                if to not in visited:
                    heapq.heappush(edges, (weight, v, to))
    return mst

# 12. Graph Cycle Detection (Directed)
# Time Complexity: O(V+E) | Space Complexity: O(V)
def has_cycle_directed(graph):
    visited = set()
    rec_stack = set()
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

# ========================
# Dynamic Programming (10)
# ========================
"""
This section contains 10 dynamic programming problems. 
Includes Fibonacci sequence, longest increasing subsequence, 
longest common subsequence, 0/1 knapsack, coin change, 
grid-based DP, and more. Focuses on optimizing recursive solutions 
and recognizing overlapping subproblems.
"""

# 1. Fibonacci Sequence
# Definition: Return the nth Fibonacci number.
# Time Complexity: O(n) | Space Complexity: O(n)
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0]*(n+1)
    dp[1] = 1
    for i in range(2,n+1):
        dp[i] = dp[i-1]+dp[i-2]
    return dp[n]

# 2. Longest Common Subsequence (LCS)
# Time Complexity: O(m*n) | Space Complexity: O(m*n)
def lcs(s1,s2):
    m,n=len(s1),len(s2)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if s1[i-1]==s2[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# 3. 0/1 Knapsack
# Time Complexity: O(n*W) | Space Complexity: O(n*W)
def knapsack(weights, values, W):
    n=len(weights)
    dp=[[0]*(W+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for w in range(1,W+1):
            if weights[i-1]<=w:
                dp[i][w]=max(values[i-1]+dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w]=dp[i-1][w]
    return dp[n][W]

# 4. Coin Change
# Time Complexity: O(amount*n) | Space Complexity: O(amount)
def coin_change(coins, amount):
    dp=[float('inf')]*(amount+1)
    dp[0]=0
    for coin in coins:
        for x in range(coin, amount+1):
            dp[x]=min(dp[x], dp[x-coin]+1)
    return dp[amount] if dp[amount]!=float('inf') else -1

# 5. Minimum Path Sum in Grid
# Time Complexity: O(m*n) | Space Complexity: O(m*n)
def min_path_sum(grid):
    m,n=len(grid),len(grid[0])
    dp=[[0]*n for _ in range(m)]
    dp[0][0]=grid[0][0]
    for i in range(1,m): dp[i][0]=dp[i-1][0]+grid[i][0]
    for j in range(1,n): dp[0][j]=dp[0][j-1]+grid[0][j]
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j]=grid[i][j]+min(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

# 6. Longest Increasing Subsequence (LIS)
# Time Complexity: O(n^2) | Space Complexity: O(n)
def lis(nums):
    n=len(nums)
    dp=[1]*n
    for i in range(n):
        for j in range(i):
            if nums[i]>nums[j]:
                dp[i]=max(dp[i], dp[j]+1)
    return max(dp)

# 7. Edit Distance
# Time Complexity: O(m*n) | Space Complexity: O(m*n)
def edit_distance(word1, word2):
    m,n=len(word1),len(word2)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i==0: dp[i][j]=j
            elif j==0: dp[i][j]=i
            elif word1[i-1]==word2[j-1]: dp[i][j]=dp[i-1][j-1]
            else: dp[i][j]=1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# 8. Maximum Product Subarray
# Time Complexity: O(n) | Space Complexity: O(1)
def max_product_subarray(nums):
    max_prod = min_prod = res = nums[0]
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(num, max_prod*num)
        min_prod = min(num, min_prod*num)
        res = max(res, max_prod)
    return res

# 9. House Robber
# Time Complexity: O(n) | Space Complexity: O(n)
def house_robber(nums):
    if not nums: return 0
    n=len(nums)
    dp=[0]*(n+1)
    dp[1]=nums[0]
    for i in range(2,n+1):
        dp[i]=max(dp[i-1], dp[i-2]+nums[i-1])
    return dp[n]

# 10. Unique Paths
# Time Complexity: O(m*n) | Space Complexity: O(m*n)
def unique_paths(m,n):
    dp=[[1]*n for _ in range(m)]
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j]=dp[i-1][j]+dp[i][j-1]
    return dp[m-1][n-1]

# ========================
# Recursion & Backtracking (5)
# ========================

"""
This section contains 5 recursion and backtracking problems. 
Includes N-Queens, Rat in a Maze, Sudoku Solver, Generate Parentheses, 
and string permutations. Designed to practice systematic search, 
recursion logic, and constraint satisfaction.
"""

# 1. N-Queens
# Time Complexity: O(N!) | Space Complexity: O(N)
def solve_n_queens(n):
    res=[]
    def backtrack(row, diagonals, anti_diagonals, cols, board):
        if row==n:
            res.append([''.join(r) for r in board])
            return
        for col in range(n):
            diag=row-col
            anti_diag=row+col
            if col in cols or diag in diagonals or anti_diag in anti_diagonals:
                continue
            cols.add(col); diagonals.add(diag); anti_diagonals.add(anti_diag)
            board[row][col]='Q'
            backtrack(row+1, diagonals, anti_diagonals, cols, board)
            board[row][col]='.'
            cols.remove(col); diagonals.remove(diag); anti_diagonals.remove(anti_diag)
    backtrack(0,set(),set(),set(),[['.']*n for _ in range(n)])
    return res

# 2. Generate Parentheses
# Time Complexity: Catalan Number O(4^n/sqrt(n)) | Space Complexity: O(n)
def generate_parentheses(n):
    res=[]
    def backtrack(s='', left=0, right=0):
        if len(s)==2*n:
            res.append(s)
            return
        if left<n: backtrack(s+'(', left+1, right)
        if right<left: backtrack(s+')', left, right+1)
    backtrack()
    return res

# 3. Subsets
# Time Complexity: O(2^n) | Space Complexity: O(n)
def subsets(nums):
    res=[]
    def backtrack(path, index):
        res.append(path)
        for i in range(index,len(nums)):
            backtrack(path+[nums[i]], i+1)
    backtrack([],0)
    return res

# 4. Word Search
# Time Complexity: O(M*4^L) | Space Complexity: O(L)
def exist(board, word):
    ROWS,COLS=len(board),len(board[0])
    def dfs(r,c,i):
        if i==len(word): return True
        if r<0 or c<0 or r>=ROWS or c>=COLS or board[r][c]!=word[i]: return False
        tmp=board[r][c]; board[r][c]='#'
        res=dfs(r+1,c,i+1) or dfs(r-1,c,i+1) or dfs(r,c+1,i+1) or dfs(r,c-1,i+1)
        board[r][c]=tmp
        return res
    for r in range(ROWS):
        for c in range(COLS):
            if dfs(r,c,0): return True
    return False

# 5. Combination Sum
# Time Complexity: O(2^t) | Space Complexity: O(t) where t = target
def combination_sum(candidates,target):
    res=[]
    def backtrack(path, start, total):
        if total==target:
            res.append(list(path))
            return
        if total>target: return
        for i in range(start,len(candidates)):
            path.append(candidates[i])
            backtrack(path,i,total+candidates[i])
            path.pop()
    backtrack([],0,0)
    return res

# ========================
# Miscellaneous / Advanced (10)
# ========================
"""
This section contains 10 advanced problems. 
Covers LRU cache design, sliding window maximum, median from stream, 
top K frequent elements, word ladder, matrix manipulations, 
and other challenging problems. Designed to combine multiple DSA concepts 
and strengthen problem-solving skills.
"""

# 1. LRU Cache
# Time Complexity: O(1) per operation | Space Complexity: O(capacity)
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cache=OrderedDict()
        self.capacity=capacity
    def get(self,key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self,key,value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key]=value
        if len(self.cache)>self.capacity:
            self.cache.popitem(last=False)

# 2. Serialize / Deserialize N-ary Tree
# Time Complexity: O(n) | Space Complexity: O(n)
class NTreeNode:
    def __init__(self,val):
        self.val=val
        self.children=[]
def serialize_nary(root):
    res=[]
    def dfs(node):
        if not node: return
        res.append(str(node.val))
        res.append(str(len(node.children)))
        for child in node.children: dfs(child)
    dfs(root)
    return ','.join(res)

def deserialize_nary(data):
    vals=iter(data.split(','))
    def dfs():
        val=int(next(vals))
        size=int(next(vals))
        node=NTreeNode(val)
        for _ in range(size): node.children.append(dfs())
        return node
    return dfs()

# 3. Median of Two Sorted Arrays
# Time Complexity: O(log(min(n,m))) | Space Complexity: O(1)
def find_median_sorted_arrays(nums1, nums2):
    if len(nums1)>len(nums2): nums1,nums2=nums2,nums1
    x,y=len(nums1),len(nums2)
    low,high=0,x
    while low<=high:
        partitionX=(low+high)//2
        partitionY=(x+y+1)//2-partitionX
        maxX=float('-inf') if partitionX==0 else nums1[partitionX-1]
        minX=float('inf') if partitionX==x else nums1[partitionX]
        maxY=float('-inf') if partitionY==0 else nums2[partitionY-1]
        minY=float('inf') if partitionY==y else nums2[partitionY]
        if maxX<=minY and maxY<=minX:
            if (x+y)%2==0: return (max(maxX,maxY)+min(minX,minY))/2
            else: return max(maxX,maxY)
        elif maxX>minY: high=partitionX-1
        else: low=partitionX+1

# 4. Trapping Rain Water
# Time Complexity: O(n) | Space Complexity: O(n)
def trap(height):
    n=len(height)
    if n==0: return 0
    left,right=0,n-1
    left_max,right_max=height[left],height[right]
    water=0
    while left<right:
        if left_max<right_max:
            left+=1
            left_max=max(left_max,height[left])
            water+=max(0,left_max-height[left])
        else:
            right-=1
            right_max=max(right_max,height[right])
            water+=max(0,right_max-height[right])
    return water

# 5. Sliding Window Maximum (Revisited)
# Already implemented above

# 6. Word Ladder
# Time Complexity: O(N*M*26) | Space Complexity: O(N*M)
from collections import deque
def ladder_length(beginWord, endWord, wordList):
    wordSet=set(wordList)
    if endWord not in wordSet: return 0
    queue=deque([(beginWord,1)])
    while queue:
        word,steps=queue.popleft()
        if word==endWord: return steps
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word=word[:i]+c+word[i+1:]
                if next_word in wordSet:
                    wordSet.remove(next_word)
                    queue.append((next_word,steps+1))
    return 0

# 7. Decode Ways
# Time Complexity: O(n) | Space Complexity: O(n)
def num_decodings(s):
    if not s: return 0
    n=len(s)
    dp=[0]*(n+1)
    dp[0]=1
    dp[1]=0 if s[0]=='0' else 1
    for i in range(2,n+1):
        if s[i-1]!='0': dp[i]+=dp[i-1]
        if 10<=int(s[i-2:i])<=26: dp[i]+=dp[i-2]
    return dp[n]

# 8. Maximum Sum Rectangle in 2D Matrix
# Time Complexity: O(rows^2*cols) | Space Complexity: O(cols)
def max_sum_rectangle(matrix):
    if not matrix: return 0
    rows,cols=len(matrix),len(matrix[0])
    res=float('-inf')
    for top in range(rows):
        temp=[0]*cols
        for bottom in range(top,rows):
            for i in range(cols):
                temp[i]+=matrix[bottom][i]
            # Apply Kadane's 1D
            curr_sum=max_sum=temp[0]
            for val in temp[1:]:
                curr_sum=max(val,curr_sum+val)
                max_sum=max(max_sum,curr_sum)
            res=max(res,max_sum)
    return res

# 9. Maximum Rectangle of 1s in Binary Matrix
# Time Complexity: O(rows*cols) | Space Complexity: O(cols)
def maximal_rectangle(matrix):
    if not matrix: return 0
    cols=len(matrix[0])
    heights=[0]*cols
    max_area=0
    for row in matrix:
        for i in range(cols):
            heights[i]=heights[i]+1 if row[i]=='1' else 0
        # Compute largest rectangle in histogram
        stack=[]
        for i,h in enumerate(heights+[0]):
            while stack and heights[stack[-1]]>=h:
                height=heights[stack.pop()]
                width=i if not stack else i-stack[-1]-1
                max_area=max(max_area,height*width)
            stack.append(i)
    return max_area

# 10. Sliding Window Median
# Time Complexity: O(n log k) | Space Complexity: O(k)
import heapq
def median_sliding_window(nums,k):
    minH,maxH=[],[]
    res=[]
    for i,num in enumerate(nums):
        heapq.heappush(maxH,-num)
        heapq.heappush(minH,-heapq.heappop(maxH))
        if len(minH)>len(maxH): heapq.heappush(maxH,-heapq.heappop(minH))
        if i>=k-1: res.append(-maxH[0] if k%2 else (-maxH[0]+minH[0])/2)
        if i>=k-1:
            out=nums[i-k+1]
            if out<=-maxH[0]:
                maxH.remove(-out)
                heapq.heapify(maxH)
            else:
                minH.remove(out)
                heapq.heapify(minH)
    return res


#🎯✌🏻🔥❤️