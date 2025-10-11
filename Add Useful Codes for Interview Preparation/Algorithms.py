"""
Algorithms Python Reference
Author: Mohit Kourav
Description:
This file contains implementations of common algorithms and data structures
with headings, definitions, explanations, example usage, and time/space complexity.
"""

# --------------------
# ALGORITHMS
# --------------------

# --------------------
# SORTING ALGORITHMS
# --------------------

# 1. Bubble Sort
# Definition: Simple comparison-based sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in wrong order.
# Time Complexity: O(n^2) | Space Complexity: O(1)
# Example Usage: arr = [5,2,9]; bubble_sort(arr)

def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


# 2. Selection Sort
# Definition: Finds the minimum element repeatedly and moves it to the correct sorted position.
# Time Complexity: O(n^2) | Space Complexity: O(1)
# Example Usage: arr = [64,25,12]; selection_sort(arr)

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# 3. Insertion Sort
# Definition: Builds the sorted array one element at a time by inserting each element into its correct position.
# Time Complexity: O(n^2) | Space Complexity: O(1)
# Example Usage: arr = [12,11,13]; insertion_sort(arr)

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >=0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr


# 4. Merge Sort
# Definition: Divide and Conquer algorithm. Splits array into halves, sorts and merges them.
# Time Complexity: O(n log n) | Space Complexity: O(n)
# Example Usage: arr = [38,27,43]; merge_sort(arr)

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr


# 5. Quick Sort
# Definition: Picks a pivot, partitions array into less/greater elements, recursively sorts partitions.
# Time Complexity: O(n log n) average, O(n^2) worst | Space Complexity: O(log n)
# Example Usage: arr = [3,6,8]; quick_sort(arr)

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


# 6. Heap Sort
# Definition: Converts array into a heap and repeatedly extracts smallest element to sort.
# Time Complexity: O(n log n) | Space Complexity: O(n)
# Example Usage: arr = [4,10,3]; heap_sort(arr)
import heapq

def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]



# --------------------
# SEARCHING ALGORITHMS
# --------------------




# 1. Linear Search
# Definition: Checks each element sequentially until target is found.
# Time Complexity: O(n) | Space Complexity: O(1)
# Example Usage: arr = [10,20,30]; linear_search(arr,30)

def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1


# 2. Binary Search
# Definition: Efficient search on sorted array by dividing search space in half.
# Time Complexity: O(log n) | Space Complexity: O(1)
# Example Usage: arr = [1,2,3,4]; binary_search(arr,3)

def binary_search(arr, target):
    left, right = 0, len(arr)-1
    while left <= right:
        mid = (left + right)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# 3. Ternary Search
# Definition: Divides sorted array into three parts and searches recursively.
# Time Complexity: O(log3 n) | Space Complexity: O(1)
# Example Usage: arr = [1,2,3,4,5,6,7]; ternary_search(arr,5)

def ternary_search(arr, target):
    left, right = 0, len(arr)-1
    while left <= right:
        third = (right-left)//3
        mid1 = left + third
        mid2 = right - third
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        if target < arr[mid1]:
            right = mid1-1
        elif target > arr[mid2]:
            left = mid2+1
        else:
            left = mid1+1
            right = mid2-1
    return -1




# --------------------
# DYNAMIC PROGRAMMING ALGORITHMS
# --------------------




# 1. Fibonacci Sequence (Top-down with Memoization)
# Definition: Computes nth Fibonacci number using recursion with caching (memoization).
# Time Complexity: O(n) | Space Complexity: O(n)
# Example Usage: fibonacci(10) -> 55

def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]


# 2. Longest Common Subsequence (LCS)
# Definition: Finds the length of the longest sequence present in both strings in the same order.
# Time Complexity: O(m*n) | Space Complexity: O(m*n)
# Example Usage: lcs('AGGTAB', 'GXTXAYB') -> 4

def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]


# 3. 0/1 Knapsack Problem
# Definition: Finds maximum value that can be put in knapsack of capacity W without splitting items.
# Time Complexity: O(n*W) | Space Complexity: O(n*W)
# Example Usage: knapsack([60,100,120], [10,20,30], 50) -> 220

def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(1, W+1):
            if weights[i-1] <= w:
                dp[i][w] = max(values[i-1]+dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][W]


# 4. Coin Change Problem
# Definition: Finds minimum number of coins required to make a given amount.
# Time Complexity: O(amount*len(coins)) | Space Complexity: O(amount)
# Example Usage: coin_change([1,2,5], 11) -> 3

def coin_change(coins, amount):
    dp = [float('inf')] * (amount+1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[amount] if dp[amount] != float('inf') else -1


# 5. Minimum Path Sum in Grid
# Definition: Finds the minimum path sum from top-left to bottom-right of a grid.
# Time Complexity: O(m*n) | Space Complexity: O(m*n)
# Example Usage: min_path_sum([[1,3,1],[1,5,1],[4,2,1]]) -> 7

def min_path_sum(grid):
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    dp = [[0]*n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    return dp[m-1][n-1]



# --------------------
# GREEDY ALGORITHMS
# --------------------

# 1. Activity Selection Problem
# Definition: Select maximum number of activities that don’t overlap given start and end times.
# Time Complexity: O(n log n) due to sorting | Space Complexity: O(n)
# Example Usage: activities = [(1,3),(2,5),(4,7)]; activity_selection(activities) -> 2

def activity_selection(activities):
    activities.sort(key=lambda x: x[1])  # sort by finish time
    count = 1
    last_end = activities[0][1]
    for i in range(1, len(activities)):
        if activities[i][0] >= last_end:
            count += 1
            last_end = activities[i][1]
    return count


# 2. Fractional Knapsack Problem
# Definition: Select fractions of items to maximize value without exceeding capacity.
# Time Complexity: O(n log n) due to sorting | Space Complexity: O(n)
# Example Usage: values = [60,100,120]; weights=[10,20,30]; W=50; fractional_knapsack(values,weights,W) -> 240

def fractional_knapsack(values, weights, W):
    index = list(range(len(values)))
    ratio = [v/w for v, w in zip(values, weights)]
    index.sort(key=lambda i: ratio[i], reverse=True)
    max_value = 0
    for i in index:
        if weights[i] <= W:
            W -= weights[i]
            max_value += values[i]
        else:
            max_value += values[i] * W/weights[i]
            break
    return max_value

# --------------------
# GRAPH ALGORITHMS
# --------------------

from collections import deque, defaultdict
import heapq

# 1. Breadth-First Search (BFS)
# Definition: Traverses graph level by level using queue.
# Time Complexity: O(V+E) | Space Complexity: O(V)
# Example Usage: bfs(graph, 0)

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            queue.extend(set(graph[vertex]) - visited)
    return order


# 2. Depth-First Search (DFS)
# Definition: Traverses graph using recursion (or stack) as deep as possible first.
# Time Complexity: O(V+E) | Space Complexity: O(V)
# Example Usage: dfs(graph, 0)

def dfs(graph, start, visited=None, order=None):
    if visited is None:
        visited = set()
    if order is None:
        order = []
    visited.add(start)
    order.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, order)
    return order


# 3. Dijkstra's Shortest Path
# Definition: Finds shortest path from source to all vertices in weighted graph.
# Time Complexity: O((V+E) log V) | Space Complexity: O(V)
# Example Usage: dijkstra(graph, 0)

def dijkstra(graph, start):
    heap = [(0, start)]
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(heap, (dist[v], v))
    return dist


# 4. Kruskal's Minimum Spanning Tree (MST)
# Definition: Finds MST by sorting edges and adding them without forming cycles.
# Time Complexity: O(E log E) | Space Complexity: O(V)
# Example Usage: kruskal(edges, num_vertices)

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        self.parent[self.find(u)] = self.find(v)

def kruskal(edges, n):
    edges.sort(key=lambda x: x[2])  # sort by weight
    ds = DisjointSet(n)
    mst = []
    for u, v, w in edges:
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst.append((u, v, w))
    return mst


# 5. Prim's Minimum Spanning Tree (MST)
# Definition: Builds MST starting from a node, repeatedly adding smallest edge connecting visited to unvisited vertices.
# Time Complexity: O(E log V) | Space Complexity: O(V)
# Example Usage: prim(graph, 0)

def prim(graph, start):
    visited = set([start])
    edges = [(weight, start, v) for v, weight in graph[start]]
    heapq.heapify(edges)
    mst = []
    while edges:
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            for to_next, w in graph[v]:
                if to_next not in visited:
                    heapq.heappush(edges, (w, v, to_next))
    return mst

