import tkinter as tk
from tkinter import ttk
import math

class PomodoroTimer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pomodoro Timer")

        # Smaller, centered window
        window_width = 400
        window_height = 500
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(True, True)
        self.root.configure(bg="#1E1E2E")

        # Timer settings (in seconds)
        self.work_duration = 25 * 60  # 25 minutes
        self.short_break = 5 * 60     # 5 minutes
        self.long_break = 15 * 60     # 15 minutes

        # Timer state
        self.time_left = self.work_duration
        self.total_time = self.work_duration
        self.is_running = False
        self.is_break = False
        self.timer_id = None
        self.sessions_completed = 0

        self.setup_ui()
        self.draw_circle()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#1E1E2E")
        header_frame.pack(pady=10, expand=True)

        title_label = tk.Label(
            header_frame,
            text="🍅 POMODORO",
            font=("Segoe UI", 28, "bold"),
            bg="#1E1E2E",
            fg="#F38BA8"
        )
        title_label.pack()

        # Session counter
        self.session_label = tk.Label(
            self.root,
            text=f"✅ Sessions: {self.sessions_completed}",
            font=("Segoe UI", 13),
            bg="#1E1E2E",
            fg="#89DCEB"
        )
        self.session_label.pack(pady=5)

        # Timer mode label
        self.mode_label = tk.Label(
            self.root,
            text="💼 WORK TIME",
            font=("Segoe UI", 16, "bold"),
            bg="#1E1E2E",
            fg="#A6E3A1"
        )
        self.mode_label.pack(pady=5)

        # Canvas for circular progress
        self.canvas = tk.Canvas(
            self.root,
            width=250,
            height=250,
            bg="#1E1E2E",
            highlightthickness=0
        )
        self.canvas.pack(pady=10, expand=True)

        # Timer display
        self.timer_label = tk.Label(
            self.canvas,
            text="25:00",
            font=("Segoe UI", 48, "bold"),
            bg="#1E1E2E",
            fg="#CDD6F4"
        )
        self.canvas.create_window(125, 125, window=self.timer_label)

        # Control buttons frame
        button_frame = tk.Frame(self.root, bg="#1E1E2E")
        button_frame.pack(pady=10, expand=True)

        # Start/Pause button
        self.start_button = tk.Button(
            button_frame,
            text="▶ START",
            font=("Segoe UI", 13, "bold"),
            bg="#A6E3A1",
            fg="#1E1E2E",
            width=10,
            height=2,
            border=0,
            cursor="hand2",
            relief="flat",
            command=self.toggle_timer
        )
        self.start_button.grid(row=0, column=0, padx=8)

        # Reset button
        self.reset_button = tk.Button(
            button_frame,
            text="⟲ RESET",
            font=("Segoe UI", 13, "bold"),
            bg="#F38BA8",
            fg="#1E1E2E",
            width=10,
            height=2,
            border=0,
            cursor="hand2",
            relief="flat",
            command=self.reset_timer
        )
        self.reset_button.grid(row=0, column=1, padx=8)

        # Settings container
        settings_container = tk.Frame(self.root, bg="#313244", relief="flat")
        settings_container.pack(pady=10, padx=20, fill="x")

        # Settings title
        tk.Label(
            settings_container,
            text="⚙️ Settings",
            font=("Segoe UI", 12, "bold"),
            bg="#313244",
            fg="#CDD6F4"
        ).pack(pady=(8, 4))

        # Settings inputs frame
        settings_frame = tk.Frame(settings_container, bg="#313244")
        settings_frame.pack(pady=5)

        # Work duration
        tk.Label(
            settings_frame,
            text="Work:",
            font=("Segoe UI", 11),
            bg="#313244",
            fg="#BAC2DE"
        ).grid(row=0, column=0, padx=5, sticky="e")

        self.work_entry = tk.Entry(
            settings_frame,
            width=5,
            font=("Segoe UI", 11),
            bg="#45475A",
            fg="#CDD6F4",
            relief="flat",
            insertbackground="#CDD6F4",
            justify="center"
        )
        self.work_entry.insert(0, "25")
        self.work_entry.grid(row=0, column=1, padx=5)

        tk.Label(
            settings_frame,
            text="min",
            font=("Segoe UI", 11),
            bg="#313244",
            fg="#6C7086"
        ).grid(row=0, column=2, padx=5, sticky="w")

        # Break duration
        tk.Label(
            settings_frame,
            text="Break:",
            font=("Segoe UI", 11),
            bg="#313244",
            fg="#BAC2DE"
        ).grid(row=0, column=3, padx=(20, 5), sticky="e")

        self.break_entry = tk.Entry(
            settings_frame,
            width=5,
            font=("Segoe UI", 11),
            bg="#45475A",
            fg="#CDD6F4",
            relief="flat",
            insertbackground="#CDD6F4",
            justify="center"
        )
        self.break_entry.insert(0, "5")
        self.break_entry.grid(row=0, column=4, padx=5)

        tk.Label(
            settings_frame,
            text="min",
            font=("Segoe UI", 11),
            bg="#313244",
            fg="#6C7086"
        ).grid(row=0, column=5, padx=5, sticky="w")

        # Apply button
        apply_button = tk.Button(
            settings_container,
            text="✓ Apply",
            font=("Segoe UI", 10, "bold"),
            bg="#89B4FA",
            fg="#1E1E2E",
            border=0,
            cursor="hand2",
            relief="flat",
            width=12,
            command=self.apply_settings
        )
        apply_button.pack(pady=(4, 8))

    def draw_circle(self):
        """Draw circular progress indicator"""
        self.canvas.delete("progress")

        # Background circle
        self.canvas.create_oval(
            15, 15, 235, 235,
            outline="#313244",
            width=12,
            tags="progress"
        )

        # Progress arc
        if self.total_time > 0:
            progress = (self.time_left / self.total_time)
            angle = 360 * progress

            if self.is_break:
                color = "#F9E2AF"  # Yellow for break
            else:
                color = "#A6E3A1"  # Green for work

            if angle > 0:
                self.canvas.create_arc(
                    15, 15, 235, 235,
                    start=90,
                    extent=angle,
                    outline=color,
                    width=12,
                    style="arc",
                    tags="progress"
                )

    def format_time(self, seconds):
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins:02d}:{secs:02d}"

    def toggle_timer(self):
        if self.is_running:
            self.pause_timer()
        else:
            self.start_timer()

    def start_timer(self):
        self.is_running = True
        self.start_button.config(text="⏸ PAUSE", bg="#F9E2AF")
        self.countdown()

    def pause_timer(self):
        self.is_running = False
        self.start_button.config(text="▶ START", bg="#A6E3A1")
        if self.timer_id:
            self.root.after_cancel(self.timer_id)

    def countdown(self):
        if self.is_running and self.time_left > 0:
            self.time_left -= 1
            self.timer_label.config(text=self.format_time(self.time_left))
            self.draw_circle()
            self.timer_id = self.root.after(1000, self.countdown)
        elif self.is_running and self.time_left == 0:
            self.timer_finished()

    def timer_finished(self):
        self.is_running = False

        # Play sound (beep)
        try:
            import winsound
            winsound.Beep(1000, 500)
        except:
            print('\a')  # Fallback beep

        if not self.is_break:
            # Work session finished
            self.sessions_completed += 1
            self.session_label.config(text=f"✅ Sessions: {self.sessions_completed}")

            # Start break
            if self.sessions_completed % 4 == 0:
                self.time_left = self.long_break
                self.total_time = self.long_break
                self.mode_label.config(text="☕ LONG BREAK", fg="#F9E2AF")
            else:
                self.time_left = self.short_break
                self.total_time = self.short_break
                self.mode_label.config(text="☕ SHORT BREAK", fg="#F9E2AF")

            self.is_break = True
        else:
            # Break finished, start work session
            self.time_left = self.work_duration
            self.total_time = self.work_duration
            self.mode_label.config(text="💼 WORK TIME", fg="#A6E3A1")
            self.is_break = False

        self.timer_label.config(text=self.format_time(self.time_left))
        self.draw_circle()
        self.start_button.config(text="▶ START", bg="#A6E3A1")

    def reset_timer(self):
        self.is_running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)

        self.is_break = False
        self.time_left = self.work_duration
        self.total_time = self.work_duration
        self.timer_label.config(text=self.format_time(self.time_left))
        self.mode_label.config(text="💼 WORK TIME", fg="#A6E3A1")
        self.draw_circle()
        self.start_button.config(text="▶ START", bg="#A6E3A1")

    def apply_settings(self):
        if not self.is_running:
            try:
                work_mins = int(self.work_entry.get())
                break_mins = int(self.break_entry.get())

                if work_mins > 0 and break_mins > 0:
                    self.work_duration = work_mins * 60
                    self.short_break = break_mins * 60
                    self.reset_timer()
            except ValueError:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = PomodoroTimer(root)
    root.mainloop()
