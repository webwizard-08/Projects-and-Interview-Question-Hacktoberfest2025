import csv
import datetime
import os

BOOKS_FILE = "Python/Library/books.csv"
LOGS_FILE = "Python/Library/logs.csv"
def load_books():
    books = []
    if os.path.exists(BOOKS_FILE):
        with open(BOOKS_FILE, "r", newline='') as file:
            reader = csv.reader(file)
            books = [row for row in reader]
    return books

def save_books(books):
    with open(BOOKS_FILE, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(books)

def display_books():
    books = load_books()
    print("\n📚 Available Books 📚\n")
    for book in books:
        print(f"ISBN: {book[0]}")
        print(f"Title: {book[1]}")
        print(f"Author: {book[2]}")
        print(f"Genre: {book[3]}")
        print(f"Availability: {'Yes' if book[4] == 'True' else 'No'}")
        print("-" * 50)

def log_transaction(isbn, title, action, borrow_date=None, due_date=None, return_date=None, fine=0):
    with open(LOGS_FILE, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            isbn, title, action, 
            borrow_date.strftime("%d-%m-%Y %H:%M:%S") if borrow_date else "N/A",
            due_date.strftime("%d-%m-%Y %H:%M:%S") if due_date else "N/A",
            return_date.strftime("%d-%m-%Y %H:%M:%S") if return_date else "N/A",
            fine
        ])

def borrow_book():
    books = load_books()
    isbn = input("Enter ISBN of the book to borrow: ").strip()

    for book in books:
        if book[0] == isbn and book[4] == 'True':  
            book[4] = 'False'  

            borrow_date = datetime.datetime.now()
            due_date = borrow_date + datetime.timedelta(days=7)

            log_transaction(book[0], book[1], "Borrowed", borrow_date, due_date)
            save_books(books)

            print(f"\n✅ You borrowed '{book[1]}'. Due date: {due_date.strftime('%d-%m-%Y')}\n")
            return

    print("\n❌ Book is not available or doesn't exist.")

def return_book():
    books = load_books()
    isbn = input("Enter ISBN of the book to return: ").strip()

    for book in books:
        if book[0] == isbn and book[4] == 'False': 
            book[4] = 'True'  

            return_date = datetime.datetime.now()
            due_date = return_date - datetime.timedelta(days=7) 
            fine = max(0, (return_date - due_date).days - 7) * 10  

            log_transaction(book[0], book[1], "Returned", None, due_date, return_date, fine)
            save_books(books)

            print(f"\n✅ You returned '{book[1]}'. Fine: ${fine}\n")
            return

    print("\n❌ Invalid ISBN or book was not borrowed.\n")

def search_books():
    books = load_books()
    keyword = input("Enter title, author, or genre to search: ").strip().lower()

    found_books = [book for book in books if keyword in book[1].lower() or keyword in book[2].lower() or keyword in book[3].lower()]

    if found_books:
        print("\n🔎 Search Results:\n")
        for book in found_books:
            print(f"ISBN: {book[0]}, Title: {book[1]}, Author: {book[2]}, Genre: {book[3]}, Available: {'Yes' if book[4] == 'True' else 'No'}")
    else:
        print("\n❌ No books found matching your search.\n")

def add_book():
    isbn = input("Enter ISBN: ").strip()
    title = input("Enter title: ").strip()
    author = input("Enter author: ").strip()
    genre = input("Enter genre: ").strip()

    books = load_books()
    books.append([isbn, title, author, genre, 'True']) 
    save_books(books)

    print(f"\n✅ Book '{title}' added successfully!\n")

def remove_book():
    books = load_books()
    isbn = input("Enter ISBN of book to remove: ").strip()

    new_books = [book for book in books if book[0] != isbn]

    if len(new_books) < len(books): 
        save_books(new_books)
        print("\n✅ Book removed successfully!\n")
    else:
        print("\n❌ Book not found.\n")

def main():
    while True:
        print("\n📚 Library Management System 📚\n")
        print("1️⃣  Display All Books")
        print("2️⃣  Search Books")
        print("3️⃣  Borrow a Book")
        print("4️⃣  Return a Book")
        print("5️⃣  Add a New Book")
        print("6️⃣  Remove a Book")
        print("7️⃣  Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            display_books()
        elif choice == "2":
            search_books()
        elif choice == "3":
            borrow_book()
        elif choice == "4":
            return_book()
        elif choice == "5":
            add_book()
        elif choice == "6":
            remove_book()
        elif choice == "7":
            print("\n👋 Thank you for using the library system!\n")
            break
        else:
            print("\n❌ Invalid option. Please try again.\n")

if __name__ == "__main__":
    main()
