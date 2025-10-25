# Question:
# You are given a partially filled 9x9 Sudoku board. Your task is to fill in the empty cells so that the board becomes a valid Sudoku solution.
#
# A valid Sudoku solution must satisfy all of the following rules:
# - Each of the digits 1–9 must occur exactly once in each row.
# - Each of the digits 1–9 must occur exactly once in each column.
# - Each of the digits 1–9 must occur exactly once in each of the nine 3x3 sub-boxes of the grid.
#
# In the input board, some cells are filled with digits ('1' to '9'), and others are empty, marked with the character '.'.
#
# Input:
# User Task
# Implement the function solve_sudoku(board) which takes a 9x9 2D character array 'board' representing a partially filled Sudoku board.
# You must fill in the empty cells (denoted by '.') such that the final board is a valid Sudoku solution.
#
# Output:
# You are not required to return or print anything. The solution will be verified based on in-place modifications made to the original 9x9 2D character array 'board'.
#
# Constraints:
# - board.length == 9
# - board[i].length == 9
# - board[i][j] is a digit or '.'
# - It is guaranteed that the input board has exactly one valid solution.
#
# Example:
# Input:
# [
# "53..7....",
# "6..195...",
# ".98....6.",
# "8...6...3",
# "4..8.3..1",
# "7...2...6",
# ".6....28.",
# "...419..5",
# "....8..79"
# ]
#
# Output:
# [
# "534678912",
# "672195348",
# "198342567",
# "859761423",
# "426853791",
# "713924856",
# "961537284",
# "287419635",
# "345286179"
# ]

# Solution:
def solve_sudoku(board):
    def is_valid(r, c, ch):
        # Check row and column
        for i in range(9):
            if board[r][i] == ch or board[i][c] == ch:
                return False
        # Check 3x3 sub-box
        start_row, start_col = 3 * (r // 3), 3 * (c // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == ch:
                    return False
        return True

    def backtrack():
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for ch in map(str, range(1, 10)):
                        if is_valid(r, c, ch):
                            board[r][c] = ch
                            if backtrack():
                                return True
                            board[r][c] = '.'  # backtrack
                    return False
        return True

    backtrack()


# Test Cases
test_cases = [
    [
        list("53..7...."),
        list("6..195..."),
        list(".98....6."),
        list("8...6...3"),
        list("4..8.3..1"),
        list("7...2...6"),
        list(".6....28."),
        list("...419..5"),
        list("....8..79")
    ],
    [
        list("..9748..."),
        list("7........"),
        list(".2.1.9..."),
        list("..7...24."),
        list(".64.1.59."),
        list(".98...3.."),
        list("...8.3.2."),
        list("........6"),
        list("...2759..")
    ],
    [
        list("1.......2"),
        list("..2..3..."),
        list("..3..4..."),
        list("...4..5.."),
        list("...5..6.."),
        list("..6..7..."),
        list("..7..8..."),
        list("...8..9.."),
        list("9.......1")
    ]
]

for idx, board in enumerate(test_cases, 1):
    print(f"\nTest Case {idx}:")
    solve_sudoku(board)
    for row in board:
        print("".join(row))
