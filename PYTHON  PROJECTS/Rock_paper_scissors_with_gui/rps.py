import tkinter as tk
from tkinter import font as tkfont, messagebox
import random
from enum import Enum
from collections import Counter

class Choice(Enum):
    """Game choices enumeration"""
    ROCK = "✊"
    PAPER = "✋"
    SCISSORS = "✌️"

class Difficulty(Enum):
    """AI difficulty levels"""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class RPSGame:
    """Advanced Rock Paper Scissors Game with AI and Round Limits"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.initialize_game_state()
        self.show_game_setup()
        
    def setup_window(self):
        """Configure main window"""
        self.root.title("Rock Paper Scissors Championship")
        self.root.geometry("700x800")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        
    def initialize_game_state(self):
        """Initialize game variables"""
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.current_round = 0
        self.max_rounds = None
        self.difficulty = Difficulty.EASY
        self.player_choice = None
        self.computer_choice = None
        self.player_history = []
        self.game_active = False
        
    def show_game_setup(self):
        """Show initial setup screen"""
        self.setup_frame = tk.Frame(self.root, bg="#1a1a2e")
        self.setup_frame.pack(expand=True, fill=tk.BOTH)
        
        title_font = tkfont.Font(family="Arial", size=32, weight="bold")
        title = tk.Label(
            self.setup_frame,
            text="🎮 Rock Paper Scissors 🎮",
            font=title_font,
            bg="#1a1a2e",
            fg="#00d4ff"
        )
        title.pack(pady=40)
        
        subtitle = tk.Label(
            self.setup_frame,
            text="Championship Edition",
            font=("Arial", 16, "italic"),
            bg="#1a1a2e",
            fg="#ffffff"
        )
        subtitle.pack(pady=5)
        
        # Difficulty selection
        diff_frame = tk.Frame(self.setup_frame, bg="#0f3460", relief=tk.RIDGE, bd=3)
        diff_frame.pack(pady=30, padx=50, fill=tk.X)
        
        tk.Label(
            diff_frame,
            text="Select Difficulty Level",
            font=("Arial", 18, "bold"),
            bg="#0f3460",
            fg="#00d4ff"
        ).pack(pady=15)
        
        self.difficulty_var = tk.StringVar(value=Difficulty.EASY.value)
        
        diff_descriptions = {
            Difficulty.EASY: "🟢 Random AI - Perfect for beginners",
            Difficulty.MEDIUM: "🟡 Pattern Recognition - Learns from 3 moves",
            Difficulty.HARD: "🔴 Adaptive AI - Advanced prediction algorithm"
        }
        
        for difficulty in Difficulty:
            rb_frame = tk.Frame(diff_frame, bg="#0f3460")
            rb_frame.pack(pady=5)
            
            rb = tk.Radiobutton(
                rb_frame,
                text=diff_descriptions[difficulty],
                variable=self.difficulty_var,
                value=difficulty.value,
                font=("Arial", 13),
                bg="#0f3460",
                fg="#ffffff",
                selectcolor="#1a1a2e",
                activebackground="#0f3460",
                activeforeground="#00d4ff",
                width=40,
                anchor="w"
            )
            rb.pack(padx=20)
        
        # Round selection
        round_frame = tk.Frame(self.setup_frame, bg="#0f3460", relief=tk.RIDGE, bd=3)
        round_frame.pack(pady=30, padx=50, fill=tk.X)
        
        tk.Label(
            round_frame,
            text="Select Game Mode",
            font=("Arial", 18, "bold"),
            bg="#0f3460",
            fg="#00d4ff"
        ).pack(pady=15)
        
        self.rounds_var = tk.StringVar(value="unlimited")
        
        round_options = [
            ("unlimited", "♾️ Unlimited Rounds - Play forever!"),
            ("3", "🥉 Best of 3 - Quick match"),
            ("5", "🥈 Best of 5 - Standard game"),
            ("10", "🥇 Best of 10 - Championship mode")
        ]
        
        for value, text in round_options:
            rb = tk.Radiobutton(
                round_frame,
                text=text,
                variable=self.rounds_var,
                value=value,
                font=("Arial", 13),
                bg="#0f3460",
                fg="#ffffff",
                selectcolor="#1a1a2e",
                activebackground="#0f3460",
                activeforeground="#00d4ff",
                width=40,
                anchor="w"
            )
            rb.pack(pady=5, padx=20)
        
        # Start button
        start_btn = tk.Button(
            self.setup_frame,
            text="🚀 Start Game",
            font=("Arial", 18, "bold"),
            bg="#27ae60",
            fg="#ffffff",
            activebackground="#229954",
            width=20,
            height=2,
            cursor="hand2",
            command=self.start_game
        )
        start_btn.pack(pady=40)
        
    def start_game(self):
        """Start the game with selected settings"""
        diff_name = self.difficulty_var.get()
        self.difficulty = next(d for d in Difficulty if d.value == diff_name)
        
        rounds = self.rounds_var.get()
        self.max_rounds = None if rounds == "unlimited" else int(rounds)
        
        self.game_active = True
        self.setup_frame.destroy()
        self.create_game_ui()
        
    def create_game_ui(self):
        """Create main game interface"""
        # Title
        title_font = tkfont.Font(family="Arial", size=24, weight="bold")
        title_label = tk.Label(
            self.root,
            text="🎮 Rock Paper Scissors Championship 🎮",
            font=title_font,
            bg="#1a1a2e",
            fg="#00d4ff"
        )
        title_label.pack(pady=15)
        
        # Game info
        self.create_game_info()
        
        # Scoreboard
        self.create_scoreboard()
        
        # Game area
        self.create_game_area()
        
        # Choice buttons
        self.create_choice_buttons()
        
        # Control buttons
        self.create_control_buttons()
        
    def create_game_info(self):
        """Display current game settings"""
        info_frame = tk.Frame(self.root, bg="#0f3460", relief=tk.RIDGE, bd=2)
        info_frame.pack(pady=10, padx=20, fill=tk.X)
        
        info_font = ("Arial", 11, "bold")
        
        left_frame = tk.Frame(info_frame, bg="#0f3460")
        left_frame.pack(side=tk.LEFT, expand=True, pady=8)
        
        tk.Label(
            left_frame,
            text=f"⚙️ Difficulty: {self.difficulty.value}",
            font=info_font,
            bg="#0f3460",
            fg="#ffffff"
        ).pack()
        
        right_frame = tk.Frame(info_frame, bg="#0f3460")
        right_frame.pack(side=tk.RIGHT, expand=True, pady=8)
        
        rounds_text = "♾️ Unlimited" if self.max_rounds is None else f"🎯 Best of {self.max_rounds}"
        self.round_info_label = tk.Label(
            right_frame,
            text=f"Round: 0 | {rounds_text}",
            font=info_font,
            bg="#0f3460",
            fg="#ffffff"
        )
        self.round_info_label.pack()
        
    def create_scoreboard(self):
        """Create scoreboard display"""
        scoreboard_frame = tk.Frame(self.root, bg="#0f3460", relief=tk.RIDGE, bd=3)
        scoreboard_frame.pack(pady=15, padx=20, fill=tk.X)
        
        score_font = tkfont.Font(family="Arial", size=18, weight="bold")
        
        # Wins
        wins_frame = tk.Frame(scoreboard_frame, bg="#0f3460")
        wins_frame.pack(side=tk.LEFT, expand=True, pady=12)
        tk.Label(wins_frame, text="🏆 WINS", font=("Arial", 13, "bold"), 
                bg="#0f3460", fg="#00ff00").pack()
        self.wins_label = tk.Label(wins_frame, text="0", font=score_font,
                                   bg="#0f3460", fg="#ffffff")
        self.wins_label.pack()
        
        # Draws
        draws_frame = tk.Frame(scoreboard_frame, bg="#0f3460")
        draws_frame.pack(side=tk.LEFT, expand=True, pady=12)
        tk.Label(draws_frame, text="🤝 DRAWS", font=("Arial", 13, "bold"),
                bg="#0f3460", fg="#ffff00").pack()
        self.draws_label = tk.Label(draws_frame, text="0", font=score_font,
                                    bg="#0f3460", fg="#ffffff")
        self.draws_label.pack()
        
        # Losses
        losses_frame = tk.Frame(scoreboard_frame, bg="#0f3460")
        losses_frame.pack(side=tk.LEFT, expand=True, pady=12)
        tk.Label(losses_frame, text="💔 LOSSES", font=("Arial", 13, "bold"),
                bg="#0f3460", fg="#ff0000").pack()
        self.losses_label = tk.Label(losses_frame, text="0", font=score_font,
                                     bg="#0f3460", fg="#ffffff")
        self.losses_label.pack()
        
    def create_game_area(self):
        """Create main game display area"""
        game_frame = tk.Frame(self.root, bg="#1a1a2e")
        game_frame.pack(pady=25)
        
        choice_font = tkfont.Font(family="Arial", size=70)
        
        # Player choice
        player_frame = tk.Frame(game_frame, bg="#0f3460", relief=tk.RIDGE, bd=3)
        player_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(player_frame, text="You", font=("Arial", 16, "bold"),
                bg="#0f3460", fg="#00d4ff", width=12).pack(pady=5)
        self.player_choice_label = tk.Label(
            player_frame,
            text="❓",
            font=choice_font,
            bg="#0f3460",
            fg="#ffffff",
            width=3,
            height=2
        )
        self.player_choice_label.pack(padx=15, pady=15)
        
        # VS label
        vs_label = tk.Label(
            game_frame,
            text="VS",
            font=("Arial", 24, "bold"),
            bg="#1a1a2e",
            fg="#ff6b6b"
        )
        vs_label.pack(side=tk.LEFT, padx=15)
        
        # Computer choice
        computer_frame = tk.Frame(game_frame, bg="#0f3460", relief=tk.RIDGE, bd=3)
        computer_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(computer_frame, text="Computer", font=("Arial", 16, "bold"),
                bg="#0f3460", fg="#ff6b6b", width=12).pack(pady=5)
        self.computer_choice_label = tk.Label(
            computer_frame,
            text="❓",
            font=choice_font,
            bg="#0f3460",
            fg="#ffffff",
            width=3,
            height=2
        )
        self.computer_choice_label.pack(padx=15, pady=15)
        
        # Result label
        result_font = tkfont.Font(family="Arial", size=20, weight="bold")
        self.result_label = tk.Label(
            self.root,
            text="Choose your weapon!",
            font=result_font,
            bg="#1a1a2e",
            fg="#ffffff",
            height=2
        )
        self.result_label.pack(pady=20)
        
    def create_choice_buttons(self):
        """Create Rock, Paper, Scissors buttons"""
        self.button_frame = tk.Frame(self.root, bg="#1a1a2e")
        self.button_frame.pack(pady=20)
        
        button_font = tkfont.Font(family="Arial", size=18, weight="bold")
        
        self.choice_buttons = {}
        choices = [
            (Choice.ROCK, "#e74c3c"),
            (Choice.PAPER, "#3498db"),
            (Choice.SCISSORS, "#f39c12")
        ]
        
        for choice, color in choices:
            btn = tk.Button(
                self.button_frame,
                text=f"{choice.value}\n{choice.name}",
                font=button_font,
                bg=color,
                fg="#ffffff",
                activebackground=self.darken_color(color),
                activeforeground="#ffffff",
                width=11,
                height=3,
                relief=tk.RAISED,
                bd=5,
                cursor="hand2",
                command=lambda c=choice: self.play(c)
            )
            btn.pack(side=tk.LEFT, padx=15)
            self.choice_buttons[choice] = btn
            
    def create_control_buttons(self):
        """Create control buttons"""
        control_frame = tk.Frame(self.root, bg="#1a1a2e")
        control_frame.pack(pady=20)
        
        control_font = tkfont.Font(family="Arial", size=12, weight="bold")
        
        new_game_btn = tk.Button(
            control_frame,
            text="🆕 New Game",
            font=control_font,
            bg="#9b59b6",
            fg="#ffffff",
            activebackground="#8e44ad",
            width=15,
            height=2,
            cursor="hand2",
            command=self.new_game
        )
        new_game_btn.pack(side=tk.LEFT, padx=8)
        
        reset_btn = tk.Button(
            control_frame,
            text="🔄 Reset Score",
            font=control_font,
            bg="#27ae60",
            fg="#ffffff",
            activebackground="#229954",
            width=15,
            height=2,
            cursor="hand2",
            command=self.reset_game
        )
        reset_btn.pack(side=tk.LEFT, padx=8)
        
        exit_btn = tk.Button(
            control_frame,
            text="❌ Exit",
            font=control_font,
            bg="#c0392b",
            fg="#ffffff",
            activebackground="#a93226",
            width=15,
            height=2,
            cursor="hand2",
            command=self.root.quit
        )
        exit_btn.pack(side=tk.LEFT, padx=8)
        
    def get_computer_choice(self):
        """Get computer's choice based on difficulty level"""
        if self.difficulty == Difficulty.EASY:
            return random.choice(list(Choice))
        
        if len(self.player_history) < 2:
            return random.choice(list(Choice))
        
        if self.difficulty == Difficulty.MEDIUM:
            # Pattern recognition - look at last 3 moves
            if len(self.player_history) >= 3:
                last_three = self.player_history[-3:]
                
                # Check if player is repeating
                if last_three[0] == last_three[1] == last_three[2]:
                    return self.get_counter(last_three[0])
                
                # Check for alternating pattern
                if last_three[0] == last_three[2]:
                    return self.get_counter(last_three[1])
            
            # Default to counter most common choice
            most_common = Counter(self.player_history).most_common(1)[0][0]
            return self.get_counter(most_common)
        
        if self.difficulty == Difficulty.HARD:
            # Advanced prediction with multiple strategies
            recent_moves = self.player_history[-5:] if len(self.player_history) >= 5 else self.player_history
            
            # Strategy 1: Counter the most common recent move (40% weight)
            if len(recent_moves) >= 3:
                counter = Counter(recent_moves)
                most_common = counter.most_common(1)[0][0]
                
                # Strategy 2: Detect and counter patterns (30% weight)
                if len(self.player_history) >= 4:
                    last_four = self.player_history[-4:]
                    # Rotating pattern detection
                    if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                        predicted = last_four[0]
                        if random.random() < 0.7:
                            return self.get_counter(predicted)
                
                # Strategy 3: Counter streaks (30% weight)
                if len(recent_moves) >= 2 and recent_moves[-1] == recent_moves[-2]:
                    if random.random() < 0.7:
                        return self.get_counter(recent_moves[-1])
                
                # Default to countering most common
                return self.get_counter(most_common)
            
            return random.choice(list(Choice))
    
    def get_counter(self, choice):
        """Get the choice that beats the given choice"""
        counters = {
            Choice.ROCK: Choice.PAPER,
            Choice.PAPER: Choice.SCISSORS,
            Choice.SCISSORS: Choice.ROCK
        }
        return counters[choice]
        
    def determine_winner(self, player, computer):
        """Determine the winner of the round"""
        if player == computer:
            return "draw"
        
        winning_combinations = {
            (Choice.ROCK, Choice.SCISSORS),
            (Choice.PAPER, Choice.ROCK),
            (Choice.SCISSORS, Choice.PAPER)
        }
        
        if (player, computer) in winning_combinations:
            return "win"
        return "lose"
        
    def play(self, player_choice):
        """Main game logic"""
        if not self.game_active:
            return
            
        # Disable buttons during animation
        self.disable_buttons()
        
        self.player_choice = player_choice
        self.computer_choice = self.get_computer_choice()
        self.player_history.append(player_choice)
        
        # Keep history manageable
        if len(self.player_history) > 15:
            self.player_history.pop(0)
        
        # Animate choices
        self.animate_choices()
        
    def disable_buttons(self):
        """Disable choice buttons during animation"""
        for btn in self.choice_buttons.values():
            btn.config(state=tk.DISABLED)
            
    def enable_buttons(self):
        """Enable choice buttons after animation"""
        for btn in self.choice_buttons.values():
            btn.config(state=tk.NORMAL)
        
    def animate_choices(self):
        """Animate the reveal of choices with flash effect"""
        self.animation_step = 0
        self.flash_choices()
        
    def flash_choices(self):
        """Flash animation for suspense"""
        if self.animation_step < 6:
            if self.animation_step % 2 == 0:
                self.player_choice_label.config(text="❓", fg="#ffff00")
                self.computer_choice_label.config(text="❓", fg="#ffff00")
            else:
                self.player_choice_label.config(text="❓", fg="#ffffff")
                self.computer_choice_label.config(text="❓", fg="#ffffff")
            
            self.animation_step += 1
            self.root.after(150, self.flash_choices)
        else:
            self.reveal_choices()
        
    def reveal_choices(self):
        """Reveal choices and determine winner"""
        # Reveal with highlight
        self.player_choice_label.config(text=self.player_choice.value, fg="#00ff00")
        self.computer_choice_label.config(text=self.computer_choice.value, fg="#ff6b6b")
        self.result_label.config(text="🎲 Calculating...", fg="#ffff00")
        
        self.root.after(500, self.show_result)
        
    def show_result(self):
        """Show final result"""
        result = self.determine_winner(self.player_choice, self.computer_choice)
        self.current_round += 1
        
        # Update scores
        if result == "win":
            self.wins += 1
            self.result_label.config(
                text=f"🎉 You Win! {self.player_choice.name} beats {self.computer_choice.name}!",
                fg="#00ff00"
            )
            self.flash_scoreboard(self.wins_label)
        elif result == "lose":
            self.losses += 1
            self.result_label.config(
                text=f"😢 You Lose! {self.computer_choice.name} beats {self.player_choice.name}!",
                fg="#ff0000"
            )
            self.flash_scoreboard(self.losses_label)
        else:
            self.draws += 1
            self.result_label.config(
                text="🤝 It's a Draw! Same choice!",
                fg="#ffff00"
            )
            self.flash_scoreboard(self.draws_label)
        
        self.update_scoreboard()
        self.update_round_info()
        
        # Reset colors after delay
        self.root.after(1000, self.reset_choice_colors)
        
        # Check if game is over
        if self.max_rounds and self.current_round >= self.max_rounds:
            self.root.after(2000, self.end_game)
        else:
            self.root.after(1000, self.enable_buttons)
            
    def flash_scoreboard(self, label):
        """Flash effect on score update"""  
        original_bg = label.cget("bg")
        label.config(bg="#ffff00", fg="#000000")
        self.root.after(200, lambda: label.config(bg=original_bg, fg="#ffffff"))
        
    def reset_choice_colors(self):
        """Reset choice label colors to default"""
        self.player_choice_label.config(fg="#ffffff")
        self.computer_choice_label.config(fg="#ffffff")
        
    def update_scoreboard(self):
        """Update scoreboard display"""
        self.wins_label.config(text=str(self.wins))
        self.losses_label.config(text=str(self.losses))
        self.draws_label.config(text=str(self.draws))
        
    def update_round_info(self):
        """Update round information display"""
        rounds_text = "♾️ Unlimited" if self.max_rounds is None else f"🎯 Best of {self.max_rounds}"
        self.round_info_label.config(text=f"Round: {self.current_round} | {rounds_text}")
        
    def end_game(self):
        """End game and show final results"""
        self.game_active = False
        self.disable_buttons()
        
        if self.wins > self.losses:
            final_msg = "🏆 VICTORY! You defeated the AI!"
            msg_color = "#00ff00"
        elif self.losses > self.wins:
            final_msg = "😢 DEFEAT! The AI wins this time!"
            msg_color = "#ff0000"
        else:
            final_msg = "🤝 TIE GAME! Perfectly matched!"
            msg_color = "#ffff00"
        
        self.result_label.config(text=final_msg, fg=msg_color)
        
        # Show detailed results
        messagebox.showinfo(
            "Game Over!",
            f"{final_msg}\n\n"
            f"Final Score:\n"
            f"Wins: {self.wins}\n"
            f"Losses: {self.losses}\n"
            f"Draws: {self.draws}\n\n"
            f"Total Rounds: {self.current_round}\n"
            f"Difficulty: {self.difficulty.value}\n\n"
            f"Click 'New Game' to play again!"
        )
        
    def reset_game(self):
        """Reset scores but keep game settings"""
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.current_round = 0
        self.player_history = []
        self.game_active = True
        
        self.player_choice_label.config(text="❓", fg="#ffffff")
        self.computer_choice_label.config(text="❓", fg="#ffffff")
        self.result_label.config(text="Choose your weapon!", fg="#ffffff")
        
        self.update_scoreboard()
        self.update_round_info()
        self.enable_buttons()
        
    def new_game(self):
        """Start completely new game with new settings"""
        # Clear current game UI
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Reset and show setup again
        self.initialize_game_state()
        self.show_game_setup()
        
    @staticmethod
    def darken_color(hex_color):
        """Darken a hex color for hover effect"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = max(0, r-30), max(0, g-30), max(0, b-30)
        return f'#{r:02x}{g:02x}{b:02x}'

def main():
    """Main entry point"""
    root = tk.Tk()
    game = RPSGame(root)
    root.mainloop()

if __name__ == "__main__":
    main()