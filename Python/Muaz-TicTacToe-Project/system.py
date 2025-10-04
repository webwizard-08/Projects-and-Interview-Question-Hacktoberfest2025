import os

# This file handles the fundamentals of our terminal-based game

class System:
    def __init__(self):
        # This initializes our board globally
        self.board = {
            1: '-', 2: '-', 3: '-',
            4: '-', 5: '-', 6: '-',
            7: '-', 8: '-', 9: '-',
        }

    # This displays our board in a presentable way
    def display(self, board):
        for i in range(1,8,3):
            print(f"{board[i]} | {board[i+1]} | {board[i+2]}")

    # This checks whether the current player has won or not. True means, the current player won. False means, the game has drawed.
    def is_win(self, player=None):
        if self.board[1] == self.board[2] == self.board[3] == player != "-": return True
        if self.board[4] == self.board[5] == self.board[6] == player != "-": return True
        if self.board[7] == self.board[8] == self.board[9] == player != "-": return True
        if self.board[1] == self.board[4] == self.board[7] == player != "-": return True
        if self.board[2] == self.board[5] == self.board[8] == player != "-": return True
        if self.board[3] == self.board[6] == self.board[9] == player != "-": return True
        if self.board[1] == self.board[5] == self.board[9] == player != "-": return True 
        if self.board[3] == self.board[5] == self.board[7] == player != "-": return True
        if "-" not in self.board.values(): return False

    # This prints heading messages in a presentable way
    def pprint(self, words):
        print("-"*50)
        print(words.center(50))
        print("-"*50)

    # This clears the clutter in the terminal session.
    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')
