import json
import os

TASKS_FILE = "tasks.json"

# Check if the JSON file exists
if not os.path.exists(TASKS_FILE):

    with open(TASKS_FILE, "w") as file:
        json.dump([], file)

# Load all tasks from the JSON file
def loadTasks():

    # Check if the file exists
    if os.path.exists(TASKS_FILE):
        
        with open(TASKS_FILE, "r") as file:
            try:
                return json.load(file)
            
            except json.JSONDecodeError:
                return []  # if file is empty / corrupted
    return []  # If the file is not found, return an empty list

# Save all tasks to the JSON file
def saveTasks(tasks):
    
    with open(TASKS_FILE, "w") as file:
        json.dump(tasks, file, indent = 4)

# Add a new task
def addTask(title):
    tasks = loadTasks()  # Load existing tasks
    tasks.append({"title": title, "completed": False})  # Add new task
    saveTasks(tasks)  # Save the updated list
    print(f"✅ Task Added: {title}")

# Show all tasks
def showTasks():
    
    tasks = loadTasks()

    # If no tasks are there
    if not tasks:
        print("📭 No tasks found! Try adding atleast one task.")
        return

    print("\nYour Tasks:")

    for i, task in enumerate(tasks, 1):
        status = "✅ Completed" if task["completed"] else "❌ Not Completed"
        print(f"{i}. {task['title']} - {status}")

# Mark Task that has completed
def completeTasks(index):
    
    tasks = loadTasks()

    if 0 <= index < len(tasks):  # Check valid index
        tasks[index]["completed"] = True
        saveTasks(tasks)
        print(f"🎯 Marked as completed: {tasks[index]['title']}")
    
    else:
        print("⚠️ Invalid Task Number")

# Delete task
def deleteTask(index):
    
    tasks = loadTasks()

    if 0 <= index < len(tasks):  
        removed = tasks.pop(index)
        saveTasks(tasks)
        print(f"🗑️ Task Deleted: {removed['title']}")
    
    else:
        print("⚠️ Invalid Task Number")


def main():

    while True:

        print("\n===== TASK MANAGER CLI =====")
        print("1. Add Task")
        print("2. List All Tasks")
        print("3. Complete Task")
        print("4. Delete Task")
        print("5. Exit\n")

        choice = input("Choose an option: ")

        if choice == "1":
            
            title = input("Enter Task Title: ")
            addTask(title)

        elif choice == "2":
            showTasks()

        elif choice == "3":
            
            tasks = loadTasks()
            
            if not tasks:
                print("📭 No tasks found! Try adding atleast one task")
                continue

            showTasks()
            index = int(input("Enter task number to complete: ")) - 1  # subtract 1 for correct index
            completeTasks(index)

        elif choice == "4":
            
            tasks = loadTasks()
            
            if not tasks:
                print("📭 No tasks found! Try adding atleast one task")
                continue
            
            showTasks()
            index = int(input("Enter task number to delete: ")) - 1 
            deleteTask(index)

        elif choice == "5":
            print("👋 Exiting Task Manager... Bye!")
            break

        else:
            print("⚠️ Invalid Choice, please try again")


if __name__ == "__main__":
    
    try:
        main()
    
    except KeyboardInterrupt:  
        print("\n👋 Exiting Task Manager... Bye!")
