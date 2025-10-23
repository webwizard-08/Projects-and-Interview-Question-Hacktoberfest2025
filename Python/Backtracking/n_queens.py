# Question:

# The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

# Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

# Each solution contains a distinct board configuration of the n-queens placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

# Note:

# A queen can attack horizontally, vertically, and diagonally. Therefore, no two queens should share the same row, column, or diagonal.
# Input
# User Task
# As this is a functional problem, Your task here is to implement the function solveNQueens, which takes only an integer n as its argument, representing the size of the chessboard.


# Custom Input
# The first line of the input contains a single integer n, representing the dimensions of the chessboard.
# Output
# Return an array of array of strings representing all the possible ways to place the n Queens on the chessboard so that no two queens attack each other.

# Custom Output
# Print the whole combinations of n x n chess board (sorted-manner), where queens are placed without attacking each other, where combinations are separated via separator.
# Constraints
# 1 ≤ n ≤ 9
# Example
# Sample Input
# 4
# Sample Output
# -------------------
# ..Q.
# Q...
# ...Q
# .Q..
# -------------------
# .Q..
# ...Q
# Q...
# ..Q.
# -------------------
# Explanation:
# There exist two distinct solutions to the 4 queens puzzle as shown above

#Solution:

def solveNQueens(n):
    board = [["." for _ in range(n)] for _ in range(n)]
    result = []

    def isSafe(row, col):
        # Check column above
        for i in range(row):
            if board[i][col] == "Q":
                return False
        # Check upper-left diagonal
        i, j = row, col
        while i >= 0 and j >= 0:
            if board[i][j] == "Q":
                return False
            i -= 1
            j -= 1
        # Check upper-right diagonal
        i, j = row, col
        while i >= 0 and j < n:
            if board[i][j] == "Q":
                return False
            i -= 1
            j += 1
        return True

    def backtrack(row):
        if row == n:
            result.append(["".join(r) for r in board])
            return
        for col in range(n):
            if isSafe(row, col):
                board[row][col] = "Q"
                backtrack(row + 1)
                board[row][col] = "."  # backtrack

    backtrack(0)
    return result


#Way to optimize - row + column in leftup or row - column in rightup or column in vertical - analyze it on paper 

# def solveNQueens(n):
#     # Write your code here
#     board = [["."]*n for _ in range(n)]
#     ans = []
#     vertical = set()
#     left_up = set()
#     right_up = set()
#     def f(row):
#         if row == n:
#             ans.append(["".join(x) for x in board])
#             return 

#         for col in range(n):
#             if (col in vertical or row+col in left_up or row-col in right_up):
#                 continue 
#             else:
#                 board[row][col] = "Q"
#                 vertical.add(col)
#                 left_up.add(row+col)
#                 right_up.add(row-col)
#                 f(row+1)
#                 board[row][col] = "."
#                 vertical.remove(col)
#                 left_up.remove(row+col)
#                 right_up.remove(row-col)
    

#     f(0)
#     return ans