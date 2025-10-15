import random
import tkinter as tk
from tkinter import ttk, messagebox

MOVES = ["Rock", "Paper", "Scissors"]
EMOJI = {"Rock": "🪨", "Paper": "📄", "Scissors": "✂️"}

def decide_winner(player, cpu):
    if player == cpu:
        return "draw"
    wins = {
        "Rock": "Scissors",
        "Paper": "Rock",
        "Scissors": "Paper",
    }
    return "player" if wins[player] == cpu else "cpu"

class RPSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rock–Paper–Scissors")
        self.geometry("520x450")
        self.minsize(520, 450)
        self.configure(bg="#0f172a")
        self.player_score = 0
        self.cpu_score = 0
        self.draws = 0
        self.best_of = tk.IntVar(value=5)
        self.game_over = False

        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        title = tk.Label(self, text="Rock–Paper–Scissors", font=("Segoe UI", 18, "bold"),
                         bg="#0f172a", fg="#e2e8f0")
        title.pack(pady=(16, 8))
        controls = tk.Frame(self, bg="#0f172a")
        controls.pack(pady=6)

        ttk.Style().configure("TCombobox", fieldbackground="white")
        ttk.Style().configure("TButton", padding=6)

        tk.Label(controls, text="Match: Best of",
                 bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 10)).grid(row=0, column=0, padx=(0, 6))

        best_of_combo = ttk.Combobox(
            controls,
            textvariable=self.best_of,
            values=[1, 3, 5, 7, 9],
            width=3,
            state="readonly",
        )
        best_of_combo.grid(row=0, column=1, padx=(0, 12))
        best_of_combo.bind("<<ComboboxSelected>>", lambda _: self._reset(match_only=True))

        ttk.Button(controls, text="Reset", command=self.reset_all).grid(row=0, column=2, padx=4)
        board = tk.Frame(self, bg="#0f172a", bd=0, highlightthickness=0)
        board.pack(pady=(8, 12), fill="x")

        card = tk.Frame(board, bg="#111827", padx=16, pady=12)
        card.pack(padx=16, fill="x")

        self.player_lbl = tk.Label(card, text="You: 0", font=("Segoe UI", 14, "bold"), fg="#f8fafc", bg="#111827")
        self.cpu_lbl = tk.Label(card, text="CPU: 0", font=("Segoe UI", 14, "bold"), fg="#f8fafc", bg="#111827")
        self.draw_lbl = tk.Label(card, text="Draws: 0", font=("Segoe UI", 11), fg="#cbd5e1", bg="#111827")
        self.status_lbl = tk.Label(card, text="Make your move!",
                                   font=("Segoe UI", 12), fg="#a7f3d0", bg="#111827")

        self.player_lbl.grid(row=0, column=0, sticky="w")
        self.cpu_lbl.grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.draw_lbl.grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.status_lbl.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        btns = tk.Frame(self, bg="#0f172a")
        btns.pack(pady=6)

        self._mk_btn(btns, "Rock", "R").grid(row=0, column=0, padx=6)
        self._mk_btn(btns, "Paper", "P").grid(row=0, column=1, padx=6)
        self._mk_btn(btns, "Scissors", "S").grid(row=0, column=2, padx=6)
        summary_card = tk.Frame(self, bg="#111827", padx=16, pady=12)
        summary_card.pack(padx=16, pady=(10, 8), fill="x")

        tk.Label(summary_card, text="Last Round",
                 font=("Segoe UI", 11, "bold"), fg="#e5e7eb", bg="#111827").grid(row=0, column=0, sticky="w")

        self.last_player = tk.Label(summary_card, text="You: —", font=("Segoe UI", 12),
                                    fg="#f8fafc", bg="#111827")
        self.last_cpu = tk.Label(summary_card, text="CPU: —", font=("Segoe UI", 12),
                                 fg="#f8fafc", bg="#111827")
        self.last_result = tk.Label(summary_card, text="Result: —", font=("Segoe UI", 12, "bold"),
                                    fg="#fde68a", bg="#111827")

        self.last_player.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.last_cpu.grid(row=1, column=1, sticky="w", padx=(20, 0), pady=(6, 0))
        self.last_result.grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(6, 0))
        history_frame = tk.Frame(self, bg="#0f172a")
        history_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        tk.Label(history_frame, text="History", font=("Segoe UI", 11, "bold"),
                 fg="#cbd5e1", bg="#0f172a").pack(anchor="w", pady=(0, 6))
        self.history = tk.Listbox(history_frame, height=8)
        self.history.pack(fill="both", expand=True)
        hint = tk.Label(self, text="Shortcuts: R / P / S   •   Reset = button above",
                        font=("Segoe UI", 9), fg="#94a3b8", bg="#0f172a")
        hint.pack(pady=(0, 8))

    def _mk_btn(self, parent, move, key):
        return ttk.Button(parent, text=f"{EMOJI[move]}  {move}  [{key}]",
                          command=lambda m=move: self.play_round(m))

    def _bind_keys(self):
        self.bind("<KeyPress-r>", lambda _: self.play_round("Rock"))
        self.bind("<KeyPress-R>", lambda _: self.play_round("Rock"))
        self.bind("<KeyPress-p>", lambda _: self.play_round("Paper"))
        self.bind("<KeyPress-P>", lambda _: self.play_round("Paper"))
        self.bind("<KeyPress-s>", lambda _: self.play_round("Scissors"))
        self.bind("<KeyPress-S>", lambda _: self.play_round("Scissors"))

    def play_round(self, player_move):
        if self.game_over:
            messagebox.showinfo("Match finished", "Match already finished. Click Reset to start a new match.")
            return

        cpu_move = random.choice(MOVES)
        outcome = decide_winner(player_move, cpu_move)

        if outcome == "player":
            self.player_score += 1
            status = "You win the round!"
            status_color = "#86efac"
        elif outcome == "cpu":
            self.cpu_score += 1
            status = "CPU wins the round!"
            status_color = "#fca5a5"
        else:
            self.draws += 1
            status = "It's a draw."
            status_color = "#fde68a"

        self._update_scoreboard(status, status_color)
        self._update_last_round(player_move, cpu_move, outcome)
        self._push_history(player_move, cpu_move, outcome)
        self._check_match_over()

    def _rounds_needed_to_win(self):
        return self.best_of.get() // 2 + 1

    def _check_match_over(self):
        target = self._rounds_needed_to_win()
        if self.player_score >= target or self.cpu_score >= target:
            self.game_over = True
            winner = "You" if self.player_score > self.cpu_score else "CPU"
            self.status_lbl.config(text=f"Match over! {winner} won.",
                                   fg="#f0abfc")
            messagebox.showinfo("Match Over",
                                f"Best of {self.best_of.get()} finished.\n"
                                f"Winner: {winner}\n\n"
                                f"Score — You: {self.player_score} | CPU: {self.cpu_score} | Draws: {self.draws}")

    def _update_scoreboard(self, status, color):
        self.player_lbl.config(text=f"You: {self.player_score}")
        self.cpu_lbl.config(text=f"CPU: {self.cpu_score}")
        self.draw_lbl.config(text=f"Draws: {self.draws}")
        self.status_lbl.config(text=status, fg=color)

    def _update_last_round(self, player, cpu, outcome):
        self.last_player.config(text=f"You: {EMOJI[player]} {player}")
        self.last_cpu.config(text=f"CPU: {EMOJI[cpu]} {cpu}")
        res_text = {"player": "You", "cpu": "CPU", "draw": "Draw"}[outcome]
        self.last_result.config(text=f"Result: {res_text}")

    def _push_history(self, player, cpu, outcome):
        res = {"player": "You", "cpu": "CPU", "draw": "Draw"}[outcome]
        self.history.insert(0, f"{EMOJI[player]} {player}  vs  {EMOJI[cpu]} {cpu}   →   {res}")

    def _reset(self, match_only=False):
        self.player_score = 0
        self.cpu_score = 0
        self.draws = 0
        self.game_over = False
        self._update_scoreboard("New match! Make your move.", "#a7f3d0")
        self._update_last_round("—", "—", "draw")
        if not match_only:
            self.history.delete(0, tk.END)

    def reset_all(self):
        self._reset(match_only=False)

if __name__ == "__main__":
    try:
        app = RPSApp()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Error", str(e))