# Question:
# You are given an n x m grid made up of land (1) and sea (0).
# People can move between adjacent land cells (up, down, left, right).
# If a group of land cells can reach the boundary of the grid, they can escape.
# Your task is to count the number of land cells that are completely trapped inside
# (i.e., cannot reach the boundary no matter how you move).
#
# Input:
# Implement the function countIsolated(grid, n, m) which takes:
# - grid: List of lists of integers (0 or 1), representing the grid
# - n: Number of rows
# - m: Number of columns
#
# Output:
# Return a single integer — the number of land cells that cannot reach the boundary.
#
# Constraints:
# 1 <= n, m <= 10^4
# 1 <= n * m <= 10^4
# 0 <= grid[i][j] <= 1
#
# Example:
# Input:
# grid = [
#   [0,0,0,0],
#   [1,0,1,0],
#   [0,1,1,0],
#   [0,0,0,0]
# ], n = 4, m = 4
#
# Output: 3
# Explanation:
# 3 land cells are completely surrounded and cannot reach the boundary.

# Solution:

from collections import deque

def countIsolated(grid, n, m):
    q = deque()
    
    # Add boundary land cells to queue
    for i in range(n):
        for j in (0, m - 1):
            if 0 <= j < m and grid[i][j] == 1:
                q.append((i, j))
                grid[i][j] = 0
    for j in range(m):
        for i in (0, n - 1):
            if 0 <= i < n and grid[i][j] == 1:
                q.append((i, j))
                grid[i][j] = 0
    
    # Directions: up, down, left, right
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    
    # BFS to mark all reachable land from boundary
    while q:
        x, y = q.popleft()
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] == 1:
                grid[nx][ny] = 0
                q.append((nx, ny))
    
    # Count remaining trapped land cells
    trapped = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                trapped += 1
    
    return trapped

# Test cases
if __name__ == "__main__":
    test_cases = [
        (
            [
                [0,0,0,0],
                [1,0,1,0],
                [0,1,1,0],
                [0,0,0,0]
            ], 4, 4
        ),  # Expected output: 3
        
        (
            [
                [1,1,1],
                [1,0,1],
                [1,1,1]
            ], 3, 3
        ),  # Expected output: 0
        
        (
            [
                [0,1,0],
                [0,1,0],
                [0,1,0]
            ], 3, 3
        ),  # Expected output: 0
        
        (
            [
                [0,0,0],
                [0,1,0],
                [0,0,0]
            ], 3, 3
        )   # Expected output: 1
    ]
    
    for grid, n, m in test_cases:
        print(f"grid={grid}, n={n}, m={m} => trapped={countIsolated(grid, n, m)}")
