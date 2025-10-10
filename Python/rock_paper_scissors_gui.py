import tkinter as tk
from tkinter import ttk, messagebox
import random
from PIL import Image, ImageTk
import os

class RockPaperScissorsGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Rock Paper Scissors Game")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Game variables
        self.player_score = 0
        self.computer_score = 0
        self.round_count = 0
        
        # Choices
        self.choices = ['rock', 'paper', 'scissors']
        self.choice_emojis = {
            'rock': '🪨',
            'paper': '📄',
            'scissors': '✂️'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="🪨 Rock Paper Scissors ✂️",
            font=('Arial', 24, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Score frame
        score_frame = tk.Frame(self.root, bg='#2c3e50')
        score_frame.pack(pady=20)
        
        # Player score
        self.player_score_label = tk.Label(
            score_frame,
            text=f"Player: {self.player_score}",
            font=('Arial', 16, 'bold'),
            fg='#3498db',
            bg='#2c3e50'
        )
        self.player_score_label.pack(side=tk.LEFT, padx=20)
        
        # VS label
        vs_label = tk.Label(
            score_frame,
            text="VS",
            font=('Arial', 16, 'bold'),
            fg='#e74c3c',
            bg='#2c3e50'
        )
        vs_label.pack(side=tk.LEFT, padx=20)
        
        # Computer score
        self.computer_score_label = tk.Label(
            score_frame,
            text=f"Computer: {self.computer_score}",
            font=('Arial', 16, 'bold'),
            fg='#e74c3c',
            bg='#2c3e50'
        )
        self.computer_score_label.pack(side=tk.LEFT, padx=20)
        
        # Round counter
        self.round_label = tk.Label(
            self.root,
            text=f"Round: {self.round_count}",
            font=('Arial', 14),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        self.round_label.pack(pady=10)
        
        # Game area frame
        game_frame = tk.Frame(self.root, bg='#2c3e50')
        game_frame.pack(pady=30)
        
        # Player choice frame
        player_frame = tk.Frame(game_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        player_frame.pack(side=tk.LEFT, padx=20)
        
        player_title = tk.Label(
            player_frame,
            text="Your Choice",
            font=('Arial', 14, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        player_title.pack(pady=10)
        
        self.player_choice_label = tk.Label(
            player_frame,
            text="❓",
            font=('Arial', 60),
            bg='#34495e'
        )
        self.player_choice_label.pack(pady=20)
        
        # Computer choice frame
        computer_frame = tk.Frame(game_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        computer_frame.pack(side=tk.RIGHT, padx=20)
        
        computer_title = tk.Label(
            computer_frame,
            text="Computer Choice",
            font=('Arial', 14, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        computer_title.pack(pady=10)
        
        self.computer_choice_label = tk.Label(
            computer_frame,
            text="❓",
            font=('Arial', 60),
            bg='#34495e'
        )
        self.computer_choice_label.pack(pady=20)
        
        # Result label
        self.result_label = tk.Label(
            self.root,
            text="Choose your move!",
            font=('Arial', 18, 'bold'),
            fg='#f39c12',
            bg='#2c3e50'
        )
        self.result_label.pack(pady=20)
        
        # Choice buttons frame
        buttons_frame = tk.Frame(self.root, bg='#2c3e50')
        buttons_frame.pack(pady=20)
        
        # Rock button
        rock_btn = tk.Button(
            buttons_frame,
            text="🪨 Rock",
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            width=10,
            height=2,
            command=lambda: self.play_game('rock')
        )
        rock_btn.pack(side=tk.LEFT, padx=10)
        
        # Paper button
        paper_btn = tk.Button(
            buttons_frame,
            text="📄 Paper",
            font=('Arial', 14, 'bold'),
            bg='#2ecc71',
            fg='white',
            width=10,
            height=2,
            command=lambda: self.play_game('paper')
        )
        paper_btn.pack(side=tk.LEFT, padx=10)
        
        # Scissors button
        scissors_btn = tk.Button(
            buttons_frame,
            text="✂️ Scissors",
            font=('Arial', 14, 'bold'),
            bg='#e74c3c',
            fg='white',
            width=10,
            height=2,
            command=lambda: self.play_game('scissors')
        )
        scissors_btn.pack(side=tk.LEFT, padx=10)
        
        # Control buttons frame
        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(pady=20)
        
        # Reset button
        reset_btn = tk.Button(
            control_frame,
            text="🔄 Reset Game",
            font=('Arial', 12, 'bold'),
            bg='#95a5a6',
            fg='white',
            width=12,
            height=2,
            command=self.reset_game
        )
        reset_btn.pack(side=tk.LEFT, padx=10)
        
        # Rules button
        rules_btn = tk.Button(
            control_frame,
            text="📋 Rules",
            font=('Arial', 12, 'bold'),
            bg='#9b59b6',
            fg='white',
            width=12,
            height=2,
            command=self.show_rules
        )
        rules_btn.pack(side=tk.LEFT, padx=10)
        
        # Exit button
        exit_btn = tk.Button(
            control_frame,
            text="🚪 Exit",
            font=('Arial', 12, 'bold'),
            bg='#e67e22',
            fg='white',
            width=12,
            height=2,
            command=self.root.quit
        )
        exit_btn.pack(side=tk.LEFT, padx=10)
        
        # Game history frame
        history_frame = tk.Frame(self.root, bg='#34495e', relief=tk.SUNKEN, bd=2)
        history_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        history_title = tk.Label(
            history_frame,
            text="Game History",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        history_title.pack(pady=5)
        
        # Scrollable text widget for history
        self.history_text = tk.Text(
            history_frame,
            height=6,
            width=70,
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1',
            state=tk.DISABLED
        )
        self.history_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Scrollbar for history
        scrollbar = tk.Scrollbar(history_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.history_text.yview)
        
    def play_game(self, player_choice):
        # Update player choice display
        self.player_choice_label.config(text=self.choice_emojis[player_choice])
        
        # Computer makes random choice
        computer_choice = random.choice(self.choices)
        self.computer_choice_label.config(text=self.choice_emojis[computer_choice])
        
        # Determine winner
        result = self.determine_winner(player_choice, computer_choice)
        
        # Update scores and round count
        self.round_count += 1
        if result == "win":
            self.player_score += 1
            result_text = "🎉 You Win!"
            result_color = "#2ecc71"
        elif result == "lose":
            self.computer_score += 1
            result_text = "😞 You Lose!"
            result_color = "#e74c3c"
        else:
            result_text = "🤝 It's a Tie!"
            result_color = "#f39c12"
        
        # Update UI
        self.result_label.config(text=result_text, fg=result_color)
        self.player_score_label.config(text=f"Player: {self.player_score}")
        self.computer_score_label.config(text=f"Computer: {self.computer_score}")
        self.round_label.config(text=f"Round: {self.round_count}")
        
        # Add to history
        self.add_to_history(player_choice, computer_choice, result_text)
        
        # Check for game end
        if self.player_score >= 5 or self.computer_score >= 5:
            self.end_game()
    
    def determine_winner(self, player_choice, computer_choice):
        if player_choice == computer_choice:
            return "tie"
        elif (player_choice == "rock" and computer_choice == "scissors") or \
             (player_choice == "paper" and computer_choice == "rock") or \
             (player_choice == "scissors" and computer_choice == "paper"):
            return "win"
        else:
            return "lose"
    
    def add_to_history(self, player_choice, computer_choice, result):
        self.history_text.config(state=tk.NORMAL)
        history_entry = f"Round {self.round_count}: You chose {player_choice} 🆚 Computer chose {computer_choice} → {result}\n"
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)
    
    def end_game(self):
        if self.player_score > self.computer_score:
            winner = "🎉 Congratulations! You won the game! 🎉"
            messagebox.showinfo("Game Over", winner)
        else:
            winner = "😞 Game Over! Computer won this time!"
            messagebox.showinfo("Game Over", winner)
        
        # Ask if player wants to play again
        play_again = messagebox.askyesno("Play Again?", "Would you like to play another game?")
        if play_again:
            self.reset_game()
        else:
            self.root.quit()
    
    def reset_game(self):
        self.player_score = 0
        self.computer_score = 0
        self.round_count = 0
        
        # Reset UI
        self.player_choice_label.config(text="❓")
        self.computer_choice_label.config(text="❓")
        self.result_label.config(text="Choose your move!", fg='#f39c12')
        self.player_score_label.config(text=f"Player: {self.player_score}")
        self.computer_score_label.config(text=f"Computer: {self.computer_score}")
        self.round_label.config(text=f"Round: {self.round_count}")
        
        # Clear history
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        self.history_text.config(state=tk.DISABLED)
    
    def show_rules(self):
        rules_text = """
🪨 Rock Paper Scissors Game Rules:

1. Choose one of three options: Rock, Paper, or Scissors
2. The computer will also make a random choice
3. Determine the winner based on these rules:
   • Rock beats Scissors 🪨 > ✂️
   • Paper beats Rock 📄 > 🪨
   • Scissors beats Paper ✂️ > 📄
   • Same choice results in a tie 🤝

4. First to reach 5 points wins the game!
5. You can reset the game anytime or view this rules dialog

Good luck and have fun! 🎮
        """
        messagebox.showinfo("Game Rules", rules_text)

def main():
    # Create the main window
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
    
    # Create and run the game
    game = RockPaperScissorsGame(root)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Start the game
    root.mainloop()

if __name__ == "__main__":
    main()
