import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class TaskManager:
    def __init__(self, filename: str = "tasks.json"):
        self.filename = filename
        self.tasks = self.load_tasks()
    
    def load_tasks(self) -> List[Dict]:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as file:
                    return json.load(file)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def save_tasks(self):
        with open(self.filename, 'w') as file:
            json.dump(self.tasks, file, indent=2)
    
    def add_task(self, title: str, description: str = "", priority: str = "medium") -> bool:
        if not title.strip():
            print("Error: Task title cannot be empty!")
            return False
        
        task = {
            "id": len(self.tasks) + 1,
            "title": title.strip(),
            "description": description.strip(),
            "priority": priority.lower(),
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        self.tasks.append(task)
        self.save_tasks()
        print(f"Task '{title}' added successfully!")
        return True
    
    def list_tasks(self, status_filter: Optional[str] = None):
        if not self.tasks:
            print("No tasks found!")
            return
        
        filtered_tasks = self.tasks
        if status_filter:
            filtered_tasks = [task for task in self.tasks if task["status"] == status_filter]
        
        if not filtered_tasks:
            print(f"No {status_filter} tasks found!")
            return
        
        print(f"\n{'='*60}")
        print(f"{'TASK LIST':^60}")
        print(f"{'='*60}")
        
        for task in filtered_tasks:
            status_icon = "✓" if task["status"] == "completed" else "○"
            priority_color = {
                "high": "🔴",
                "medium": "🟡", 
                "low": "🟢"
            }.get(task["priority"], "⚪")
            
            print(f"\n{status_icon} [{task['id']}] {task['title']}")
            print(f"   Priority: {priority_color} {task['priority'].upper()}")
            print(f"   Status: {task['status'].upper()}")
            if task['description']:
                print(f"   Description: {task['description']}")
            print(f"   Created: {task['created_at'][:19]}")
            if task['completed_at']:
                print(f"   Completed: {task['completed_at'][:19]}")
    
    def complete_task(self, task_id: int) -> bool:
        for task in self.tasks:
            if task["id"] == task_id:
                if task["status"] == "completed":
                    print(f"Task {task_id} is already completed!")
                    return False
                
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                self.save_tasks()
                print(f"Task '{task['title']}' marked as completed!")
                return True
        
        print(f"Task with ID {task_id} not found!")
        return False
    
    def delete_task(self, task_id: int) -> bool:
        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                deleted_task = self.tasks.pop(i)
                self.save_tasks()
                print(f"Task '{deleted_task['title']}' deleted successfully!")
                return True
        
        print(f"Task with ID {task_id} not found!")
        return False
    
    def update_task(self, task_id: int, title: Optional[str] = None, 
                   description: Optional[str] = None, priority: Optional[str] = None) -> bool:
        for task in self.tasks:
            if task["id"] == task_id:
                if title is not None:
                    task["title"] = title.strip()
                if description is not None:
                    task["description"] = description.strip()
                if priority is not None:
                    task["priority"] = priority.lower()
                
                self.save_tasks()
                print(f"Task {task_id} updated successfully!")
                return True
        
        print(f"Task with ID {task_id} not found!")
        return False
    
    def get_statistics(self):
        if not self.tasks:
            print("No tasks found!")
            return
        
        total_tasks = len(self.tasks)
        completed_tasks = len([task for task in self.tasks if task["status"] == "completed"])
        pending_tasks = total_tasks - completed_tasks
        
        high_priority = len([task for task in self.tasks if task["priority"] == "high"])
        medium_priority = len([task for task in self.tasks if task["priority"] == "medium"])
        low_priority = len([task for task in self.tasks if task["priority"] == "low"])
        
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        print(f"\n{'='*40}")
        print(f"{'TASK STATISTICS':^40}")
        print(f"{'='*40}")
        print(f"Total Tasks: {total_tasks}")
        print(f"Completed: {completed_tasks}")
        print(f"Pending: {pending_tasks}")
        print(f"Completion Rate: {completion_rate:.1f}%")
        print(f"\nPriority Distribution:")
        print(f"  High: {high_priority}")
        print(f"  Medium: {medium_priority}")
        print(f"  Low: {low_priority}")

def main():
    task_manager = TaskManager()
    
    while True:
        print(f"\n{'='*50}")
        print(f"{'TASK MANAGER':^50}")
        print(f"{'='*50}")
        print("1. Add Task")
        print("2. List All Tasks")
        print("3. List Pending Tasks")
        print("4. List Completed Tasks")
        print("5. Complete Task")
        print("6. Update Task")
        print("7. Delete Task")
        print("8. Show Statistics")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            title = input("Enter task title: ")
            description = input("Enter task description (optional): ")
            priority = input("Enter priority (high/medium/low, default: medium): ").strip()
            if not priority:
                priority = "medium"
            task_manager.add_task(title, description, priority)
        
        elif choice == "2":
            task_manager.list_tasks()
        
        elif choice == "3":
            task_manager.list_tasks("pending")
        
        elif choice == "4":
            task_manager.list_tasks("completed")
        
        elif choice == "5":
            try:
                task_id = int(input("Enter task ID to complete: "))
                task_manager.complete_task(task_id)
            except ValueError:
                print("Invalid task ID! Please enter a number.")
        
        elif choice == "6":
            try:
                task_id = int(input("Enter task ID to update: "))
                print("Leave blank to keep current value")
                title = input("Enter new title (optional): ").strip() or None
                description = input("Enter new description (optional): ").strip() or None
                priority = input("Enter new priority (optional): ").strip() or None
                task_manager.update_task(task_id, title, description, priority)
            except ValueError:
                print("Invalid task ID! Please enter a number.")
        
        elif choice == "7":
            try:
                task_id = int(input("Enter task ID to delete: "))
                confirm = input(f"Are you sure you want to delete task {task_id}? (y/N): ").strip().lower()
                if confirm == 'y':
                    task_manager.delete_task(task_id)
                else:
                    print("Deletion cancelled.")
            except ValueError:
                print("Invalid task ID! Please enter a number.")
        
        elif choice == "8":
            task_manager.get_statistics()
        
        elif choice == "9":
            print("Thank you for using Task Manager!")
            break
        
        else:
            print("Invalid choice! Please enter a number between 1-9.")

if __name__ == "__main__":
    main()
