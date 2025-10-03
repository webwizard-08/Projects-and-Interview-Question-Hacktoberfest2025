import customtkinter as ctk
from customtkinter import filedialog, END

import io
import contextlib

class CodeVisualizer(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()

class CodeEditor(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry("960x960")
        self.title("Python Interpreter Visualized ~ Muaz")

        self.main_frame1 = ctk.CTkFrame(self)
        self.main_frame1.pack(padx=15, pady=20, fill="both", expand=True)

        self.main_frame2 = ctk.CTkFrame(self)
        self.main_frame2.pack(padx=15, pady=20, fill="both", expand=True)

        self.text_area = ctk.CTkTextbox(
            self.main_frame1,
            font=("Consolas", 18),
            activate_scrollbars=True
        )
        self.text_area.pack(fill="both", expand=True)

        self.output_area = ctk.CTkTextbox(
            self.main_frame2,
            font=("Consolas", 18),
            activate_scrollbars=True,
        )
        self.output_area.pack(fill="both", expand=True)

        self.bind("(",self.left_para)
        self.bind("\"",self.left_dq)
        self.bind("'",self.left_apos)

        self.runbutton = ctk.CTkButton(
            self, text="▶",
            font=("Consolas", 20),
            width=40, height=7,
            border_width=0,
            corner_radius=8,
            fg_color="#484848", #1a1a1a #484848
            hover_color="#5F5F5F",
            anchor="center",
            command=self.run_code
        )
        self.runbutton.place(x=900,y=466)

        self.savefilebutton = ctk.CTkButton(
            self, text="Save",
            font=("Consolas", 15),
            width=50, height=7,
            border_width=0,
            corner_radius=8,
            fg_color="#484848", #1a1a1a #484848
            hover_color="#5F5F5F",
            anchor="center",
            command=self.save_file
        )
        self.savefilebutton.place(x=20,y=468)

        self.openfilebutton = ctk.CTkButton(
            self, text="Open",
            font=("Consolas", 15),
            width=50, height=7,
            border_width=0,
            corner_radius=8,
            fg_color="#484848", #1a1a1a #484848
            hover_color="#5F5F5F",
            anchor="center",
            command=self.open_file
        )
        self.openfilebutton.place(x=80,y=468)

        self.path = ""

    def left_para(self,event):
        self.text_area.insert("insert",")")

    def left_dq(self,event):
        self.text_area.insert("insert","\"")
    
    def left_apos(self,event):
        self.text_area.insert("insert","'")

    def set_file_name(self,path):
        self.path = path

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "r") as file:
                content = file.read()
                self.text_area.delete("1.0", END)
                self.text_area.insert("1.0", content)
                self.set_file_name(file_path)

    def save_file(self):
        if self.path == "":
            file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py"), ("All Files", "*.*")])
        else:
            self.path = file_path
        if file_path:
            with open(file_path, "w") as file:
                content = self.text_area.get("1.0", END)
                file.write(content) 
                self.set_file_name(file_path)

    def run_code(self):
        code = self.text_area.get("1.0", END)
        output = io.StringIO()
        error = io.StringIO()

        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                exec(code)

        except Exception as e:
            error.write(str(e))

        self.output_area.configure(state="normal")
        self.output_area.delete("1.0", END)
        self.output_area.insert("1.0", output.getvalue() + error.getvalue())
        self.output_area.configure(state="disabled")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app = CodeEditor()
    app.mainloop()
