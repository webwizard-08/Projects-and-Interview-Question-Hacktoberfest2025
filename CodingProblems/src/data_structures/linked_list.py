class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, value):
        """Add a new node with given value to the end of the list"""
        if not self.head:
            self.head = Node(value)
            return
        
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(value)
    
    def delete(self, value):
        """Delete the first occurrence of value in the linked list"""
        if not self.head:
            return
        
        if self.head.value == value:
            self.head = self.head.next
            return
        
        current = self.head
        while current.next and current.next.value != value:
            current = current.next
        
        if current.next:
            current.next = current.next.next
    
    def display(self):
        """Print all values in the linked list"""
        values = []
        current = self.head
        while current:
            values.append(str(current.value))
            current = current.next
        return ' -> '.join(values) if values else 'Empty list'
