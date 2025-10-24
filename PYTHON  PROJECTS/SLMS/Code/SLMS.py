import tkinter as tk
from tkinter import ttk, messagebox
from hashlib import sha256
import pandas as pd
from ttkthemes import ThemedTk
import xlsxwriter

# Global variables
excel_file_path = "C:/Users/Vedant/Documents/GitHub/SLMS/DB/LISTS.xlsx"
sheet_name_student = "Student List"
sheet_name_book = "Book List"
student_data = None
book_data = None
book_tree = None

# Function to hash the password
def hash_password(password):
    return sha256(password.encode()).hexdigest()

def authenticate(username, password):
    # User credentials (username: hashed_password)
    user_credentials = {"username": hash_password("pass")}

    # Check if the username exists and the password is correct
    return username in user_credentials and user_credentials[username] == hash_password(password)

def read_excel(excel_file_path, sheet_name):
    try:
        return pd.read_excel(excel_file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        messagebox.showerror("File Not Found", f"{excel_file_path} not found.")
    except ValueError:  # <-- Change this line
        messagebox.showerror("Sheet Not Found", f"{sheet_name} sheet not found in the file.")
    return None

def write_excel(student_data, book_data, excel_file_path):
    try:
        with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
            student_data.to_excel(writer, sheet_name=sheet_name_student, index=False)
            book_data.to_excel(writer, sheet_name=sheet_name_book, index=False)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while writing to Excel: {e}")

def find_row_by_gr(input_gr, student_data):
    return student_data[student_data["Gr No"] == input_gr]

def find_book_by_serial(serial_no, book_data):
    return book_data[book_data["Serial No."] == serial_no]

def edit_issued_books(input_gr, new_value, student_data):
    row_index = student_data[student_data["Gr No"] == input_gr].index

    if not row_index.empty:
        new_value = str(new_value)
        row_name = row_index[0]
        col_name = "Issued Books"
        student_data.at[row_name, col_name] = new_value
    else:
        messagebox.showwarning("Warning", f"GR No. {input_gr} not found in the DataFrame.")

def display_treeview(result_row, parent_frame, existing_value):
    tree = ttk.Treeview(parent_frame, columns=list(result_row.columns), show="headings")

    for col in result_row.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    student_data.loc[result_row.index, "Issued Books"] = student_data.loc[result_row.index, "Issued Books"].astype(str)

    for i, row in result_row.iterrows():
        tree.insert("", i, values=tuple(row))

    tree.pack(expand=True, fill="both")

    label_new_value = ttk.Label(parent_frame, text=f"Enter Serial No.")
    label_new_value.pack(pady=5)
    
    existing_value = result_row["Issued Books"].iloc[0]
    entry_new_value = ttk.Entry(parent_frame)
    entry_new_value.pack(pady=5)

    button_add_book = ttk.Button(parent_frame, text="Issue Book", command=lambda: add_book_to_student(result_row, entry_new_value.get()))
    button_add_book.pack(pady=5)

    button_remove_book = ttk.Button(parent_frame, text="Re-Entry Book", command=lambda: remove_book_from_student(result_row, entry_new_value.get()))
    button_remove_book.pack(pady=5)

    button_apply_edit = ttk.Button(parent_frame, text="Apply Edit", command=lambda: apply_edit_with_new_value(result_row, entry_new_value.get()))
    button_apply_edit.pack(pady=10)


def open_book_list_window():
    global book_list_window, book_tree, search_entry  # Declare global variables

    book_list_window = ThemedTk(theme="breeze")
    book_list_window.title("Book List")

    # Search Entry
    search_label = ttk.Label(book_list_window, text="Search by Name/Serial No.:")
    search_label.pack(pady=5)

    search_entry = ttk.Entry(book_list_window)
    search_entry.pack()

    search_button = ttk.Button(book_list_window, text="Search", command=search_books)
    search_button.pack(pady=5)

    # Show Issue History Button
    show_history_button = ttk.Button(book_list_window, text="Show Issue History", command=show_issue_history)
    show_history_button.pack(pady=5)

    # Treeview
    book_tree = ttk.Treeview(book_list_window, columns=["Serial No.", "Book Name", "Status", "Records", "Issue History"], show="headings")

    # Assuming 'book_data' is a DataFrame with columns: 'Serial No.', 'Book Name', 'Status', 'Records', 'Issue History'
    for col in book_tree["columns"]:
        book_tree.heading(col, text=col)
        book_tree.column(col, anchor="center")

    for i, row in book_data.iterrows():
        book_tree.insert("", i, values=tuple(row))

    book_tree.pack(expand=True, fill="both")

    book_list_window.mainloop()

def show_issue_history():
    selected_item = book_tree.selection()
    if not selected_item:
        messagebox.showwarning("Warning", "Please select a book.")
        return

    selected_item = selected_item[0]
    selected_item_values = book_tree.item(selected_item, "values")
    print(f"Selected Item Values: {selected_item_values}")  # Add this line for debugging

    if not selected_item_values:
        messagebox.showwarning("Warning", "No values found for the selected item.")
        return

    # Convert the Serial No. to an integer
    serial_no = int(selected_item_values[0])

    book_row = find_book_by_serial(serial_no, book_data)

    if book_row.empty:
        messagebox.showwarning("Warning", f"No information found for Serial No. {serial_no}.")
        return

    # Check if "Issue History" column exists
    if "Issue History" not in book_row.columns:
        messagebox.showwarning("Warning", "No 'Issue History' column found in the DataFrame.")
        return

    try:
        issue_history = book_row["Issue History"].iloc[0]

        # Replace commas with newline characters
        issue_history = issue_history.replace(", ", "\n")

        # Create a new window to display the Issue History
        history_window = tk.Toplevel()
        history_window.title("Issue History")

        text_widget = tk.Text(history_window, wrap=tk.WORD)
        text_widget.insert(tk.END, issue_history)
        text_widget.pack(expand=True, fill="both")

    except Exception as e:
        print(f"Error: {e}")  # Add this line for debugging
        messagebox.showerror("Error", f"An error occurred while processing 'Issue History': {e}")

        
auth_window = None
root = None
book_list_window = None

def add_book(input_gr, serial_no, student_data, book_data):
    student_index = student_data[student_data["Gr No"] == input_gr].index
    book_index = book_data[book_data["Serial No."] == serial_no].index

    if not student_index.empty and not book_index.empty:
        student_name = gr_student_dict[input_gr]
        book_name = serial_book_dict[serial_no]

        # Check if the book is already issued
        if book_data.at[book_index[0], "Status"] == "Issued":
            messagebox.showwarning("Warning", f"The book '{book_name}' is already issued to a student.")
            return

        # Add book to student's issued books
        current_books = student_data.at[student_index[0], "Issued Books"]
        updated_books = f"{current_books}, {book_name}" if current_books else book_name
        student_data.at[student_index[0], "Issued Books"] = updated_books

        # Update Records section
        book_data.at[book_index[0], "Records"] = f"{input_gr} ({student_name})"

        # Update Status to Issued
        book_data.at[book_index[0], "Status"] = "Issued"

        # Update Issue History
        issue_history = book_data.at[book_index[0], "Issue History"]
        current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        updated_issue_history = f"{issue_history}, {student_name} ('{current_date}' --)"
        book_data.at[book_index[0], "Issue History"] = updated_issue_history

        write_excel(student_data, book_data, excel_file_path)
        baseprint()

def remove_book(input_gr, serial_no, student_data, book_data):
    student_index = student_data[student_data["Gr No"] == input_gr].index
    book_index = book_data[book_data["Serial No."] == serial_no].index

    if not student_index.empty and not book_index.empty:
        student_name = gr_student_dict[input_gr]
        book_name = serial_book_dict[serial_no]

        # Remove book from student's issued books
        current_books = student_data.at[student_index[0], "Issued Books"]
        updated_books = ", ".join(book.strip() for book in current_books.split(",") if book.strip() != book_name)
        student_data.at[student_index[0], "Issued Books"] = updated_books

        # Update Records section
        book_data.at[book_index[0], "Records"] = ""

        # Update Status to Not Issued
        book_data.at[book_index[0], "Status"] = "Not Issued"

        # Update Issue History
        issue_history = book_data.at[book_index[0], "Issue History"]
        current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        updated_issue_history = f"{issue_history[:-1]} -- '{current_date}')"
        book_data.at[book_index[0], "Issue History"] = updated_issue_history

        write_excel(student_data, book_data, excel_file_path)
        baseprint()


def add_book_to_student(result_row, serial_no_entry_value):
    try:
        input_gr = int(result_row["Gr No"].iloc[0])
        serial_no = int(serial_no_entry_value)
        add_book(input_gr, serial_no, student_data, book_data)
    except ValueError:
        messagebox.showerror("Error", "Invalid GR No. or Serial No.")

def remove_book_from_student(result_row, serial_no_entry_value):
    try:
        input_gr = int(result_row["Gr No"].iloc[0])
        serial_no = int(serial_no_entry_value)
        remove_book(input_gr, serial_no, student_data, book_data)
    except ValueError:
        messagebox.showerror("Error", "Invalid GR No. or Serial No.")

def search_books():
    global book_tree, search_entry  # Declare global variables
    search_text = search_entry.get().strip().lower()

    # Clear existing items in the Treeview
    for item in book_tree.get_children():
        book_tree.delete(item)

    # Display matching records
    for i, row in book_data.iterrows():
        if search_text in str(row["Serial No."]).lower() or search_text in str(row["Book Name"]).lower():
            book_tree.insert("", i, values=tuple(row))
            
def apply_edit_with_new_value(result_row, new_value):
    input_gr = result_row["Gr No"].values[0]
    edit_issued_books(input_gr, new_value, student_data)
    write_excel(student_data, book_data, excel_file_path)
    baseprint()

def baseprint():
    global gr, entry_issued_books, result_frame, prompt_label  # Declare global variables
    try:
        input_gr = int(gr.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid integer for GR No.")
        return

    new_issued_books_value = entry_issued_books.get()

    result_row = find_row_by_gr(input_gr, student_data)

    if not result_row.empty:
        for widget in result_frame.winfo_children():
            widget.destroy()

        display_treeview(result_row, result_frame, new_issued_books_value)
    else:
        prompt_label.config(text=f"GR No. {input_gr} does not exist in the Excel sheet.")

def apply_edit():
    global gr_to_edit, new_issued_books_var, edit_frame, prompt_label  # Declare global variables
    input_gr = int(gr_to_edit.get())
    new_issued_books_value = new_issued_books_var.get()
    edit_issued_books(input_gr, new_issued_books_value, student_data)
    edit_frame.pack_forget()
    prompt_label.config(text=f"GR No. {input_gr} has been updated with new Issued Books: {new_issued_books_value}.")

auth_window = None
root = None

def main():
    global auth_window, student_data, book_data, gr_student_dict, serial_book_dict  # Declare global variables
    student_data = read_excel(excel_file_path, sheet_name_student)
    book_data = read_excel(excel_file_path, sheet_name_book)

    serial_book_dict = dict(zip(book_data["Serial No."], book_data["Book Name"]))
    gr_student_dict = dict(zip(student_data["Gr No"], student_data["Name"]))
    
    if student_data is None or book_data is None:
        return

    auth_window = ThemedTk(theme="breeze")
    auth_window.title("Authentication")

    auth_username_label = ttk.Label(auth_window, text="Username:")
    auth_username_label.pack()

    auth_username_entry = ttk.Entry(auth_window)
    auth_username_entry.pack()

    auth_password_label = ttk.Label(auth_window, text="Password:")
    auth_password_label.pack()

    auth_password_entry = ttk.Entry(auth_window, show="*")
    auth_password_entry.pack()

    auth_login_button = ttk.Button(auth_window, text="Login", command=lambda: on_auth_login(auth_username_entry.get(), auth_password_entry.get()))
    auth_login_button.pack()

    auth_window.mainloop()

def on_auth_login(entered_username, entered_password):
    global auth_window, root, book_data  # Declare global variables
    if authenticate(entered_username, entered_password):
        auth_window.destroy()
        initialize_main_window()
        open_book_list_window()
    else:
        messagebox.showerror("Authentication Failed", "Invalid username or password")
        
def initialize_main_window():
    global root, gr, entry_issued_books, result_frame, prompt_label, gr_to_edit, new_issued_books_var, edit_frame  # Declare global variables
    root = ThemedTk(theme="breeze")
    root.title("School Library Management System")

    label_enter_gr = ttk.Label(root, text="Enter GR No.:")
    label_enter_gr.pack(pady=5)

    gr = ttk.Entry(root)
    gr.pack()

    submit = ttk.Button(root, text="Submit", command=baseprint)
    submit.pack()

    prompt_label = ttk.Label(root, text="")
    prompt_label.pack()

    result_frame = ttk.Frame(root)
    result_frame.pack()

    edit_frame = ttk.Frame(root)
    gr_to_edit = tk.StringVar()
    new_issued_books_var = tk.StringVar()

    label_gr_to_edit = ttk.Label(edit_frame, text="Edit Issued Books for GR No.:")
    label_gr_to_edit.grid(row=0, column=0, padx=5, pady=5)

    entry_gr_to_edit = ttk.Entry(edit_frame, textvariable=gr_to_edit, state="readonly")
    entry_gr_to_edit.grid(row=0, column=1, padx=5, pady=5)

    label_new_issued_books = ttk.Label(edit_frame, text="New Issued Books:")
    label_new_issued_books.grid(row=1, column=0, padx=5, pady=5)

    entry_issued_books = ttk.Entry(edit_frame, textvariable=new_issued_books_var)
    entry_issued_books.grid(row=1, column=1, padx=5, pady=5)

    button_apply_edit = ttk.Button(edit_frame, text="Manual Edit", command=apply_edit)
    button_apply_edit.grid(row=2, column=0, columnspan=2, pady=10)

if __name__ == "__main__":
    main()