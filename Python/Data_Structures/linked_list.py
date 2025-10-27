"""
Comprehensive Linked List Implementation with Common Interview Problems

This module implements various types of linked lists and common interview problems.
Essential for technical interviews at FAANG companies.

Time/Space complexity analysis included for each operation.
"""

class ListNode:
    """Node for singly linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __str__(self):
        return str(self.val)

class DoublyListNode:
    """Node for doubly linked list."""
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class SinglyLinkedList:
    """
    Singly Linked List implementation with common operations.
    """
    
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        """Add element to end. Time: O(n), Space: O(1)"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, val):
        """Add element to beginning. Time: O(1), Space: O(1)"""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, val):
        """Delete first occurrence. Time: O(n), Space: O(1)"""
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next and current.next.val != val:
            current = current.next
        
        if current.next:
            current.next = current.next.next
            self.size -= 1
            return True
        return False
    
    def find(self, val):
        """Find element. Time: O(n), Space: O(1)"""
        current = self.head
        while current:
            if current.val == val:
                return current
            current = current.next
        return None
    
    def reverse(self):
        """Reverse linked list iteratively. Time: O(n), Space: O(1)"""
        prev = None
        current = self.head
        
        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp
        
        self.head = prev
    
    def reverse_recursive(self, node=None):
        """Reverse linked list recursively. Time: O(n), Space: O(n)"""
        if node is None:
            node = self.head
        
        if not node or not node.next:
            self.head = node
            return node
        
        reversed_head = self.reverse_recursive(node.next)
        node.next.next = node
        node.next = None
        return reversed_head
    
    def has_cycle(self):
        """Floyd's cycle detection. Time: O(n), Space: O(1)"""
        if not self.head:
            return False
        
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
    
    def find_middle(self):
        """Find middle node using two pointers. Time: O(n), Space: O(1)"""
        if not self.head:
            return None
        
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    
    def remove_nth_from_end(self, n):
        """Remove nth node from end. Time: O(n), Space: O(1)"""
        dummy = ListNode(0)
        dummy.next = self.head
        first = second = dummy
        
        # Move first n+1 steps ahead
        for _ in range(n + 1):
            first = first.next
        
        # Move both until first reaches end
        while first:
            first = first.next
            second = second.next
        
        # Remove nth node
        second.next = second.next.next
        self.head = dummy.next
        self.size -= 1
    
    def merge_sorted(self, other_list):
        """Merge two sorted linked lists. Time: O(m+n), Space: O(1)"""
        dummy = ListNode(0)
        current = dummy
        
        l1, l2 = self.head, other_list.head
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        current.next = l1 or l2
        self.head = dummy.next
    
    def display(self):
        """Display linked list."""
        result = []
        current = self.head
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result) + " -> None"

class DoublyLinkedList:
    """
    Doubly Linked List implementation.
    """
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, val):
        """Add to end. Time: O(1), Space: O(1)"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    def prepend(self, val):
        """Add to beginning. Time: O(1), Space: O(1)"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
    
    def delete(self, val):
        """Delete first occurrence. Time: O(n), Space: O(1)"""
        current = self.head
        
        while current:
            if current.val == val:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                self.size -= 1
                return True
            current = current.next
        return False

# Interview Problem Functions
def intersection_of_two_lists(headA, headB):
    """
    Find intersection point of two linked lists.
    LeetCode 160 - Intersection of Two Linked Lists
    Time: O(m + n), Space: O(1)
    """
    if not headA or not headB:
        return None
    
    pA, pB = headA, headB
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    
    return pA

def add_two_numbers(l1, l2):
    """
    Add two numbers represented as linked lists.
    LeetCode 2 - Add Two Numbers
    Time: O(max(m, n)), Space: O(max(m, n))
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        
        current.next = ListNode(total % 10)
        current = current.next
        
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next

def palindrome_linked_list(head):
    """
    Check if linked list is palindrome.
    LeetCode 234 - Palindrome Linked List
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return True
    
    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    prev = None
    while slow:
        next_temp = slow.next
        slow.next = prev
        prev = slow
        slow = next_temp
    
    # Compare both halves
    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    
    return True

def copy_random_list(head):
    """
    Deep copy linked list with random pointer.
    LeetCode 138 - Copy List with Random Pointer
    Time: O(n), Space: O(n)
    """
    if not head:
        return None
    
    # Create mapping of original to copied nodes
    old_to_new = {}
    
    # First pass: create all nodes
    current = head
    while current:
        old_to_new[current] = ListNode(current.val)
        current = current.next
    
    # Second pass: set next and random pointers
    current = head
    while current:
        if current.next:
            old_to_new[current].next = old_to_new[current.next]
        if hasattr(current, 'random') and current.random:
            old_to_new[current].random = old_to_new[current.random]
        current = current.next
    
    return old_to_new[head]

# Test Functions
def test_linked_list_operations():
    """Test basic linked list operations."""
    print("=== Testing Singly Linked List ===")
    
    ll = SinglyLinkedList()
    
    # Test append and display
    for i in [1, 2, 3, 4, 5]:
        ll.append(i)
    print(f"After appending 1-5: {ll.display()}")
    
    # Test prepend
    ll.prepend(0)
    print(f"After prepending 0: {ll.display()}")
    
    # Test find middle
    middle = ll.find_middle()
    print(f"Middle node: {middle.val}")
    
    # Test reverse
    ll.reverse()
    print(f"After reverse: {ll.display()}")
    
    # Test cycle detection
    print(f"Has cycle: {ll.has_cycle()}")
    
    # Test remove nth from end
    ll.remove_nth_from_end(2)
    print(f"After removing 2nd from end: {ll.display()}")

def test_interview_problems():
    """Test interview-specific problems."""
    print("\n=== Testing Interview Problems ===")
    
    # Test palindrome
    # Create: 1 -> 2 -> 2 -> 1
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(2)
    head.next.next.next = ListNode(1)
    
    print(f"Is palindrome (1->2->2->1): {palindrome_linked_list(head)}")
    
    # Test add two numbers
    # 342 + 465 = 807 (represented as 2->4->3 + 5->6->4 = 7->0->8)
    l1 = ListNode(2)
    l1.next = ListNode(4)
    l1.next.next = ListNode(3)
    
    l2 = ListNode(5)
    l2.next = ListNode(6)
    l2.next.next = ListNode(4)
    
    result = add_two_numbers(l1, l2)
    result_list = SinglyLinkedList()
    result_list.head = result
    print(f"342 + 465 = {result_list.display()}")

if __name__ == "__main__":
    test_linked_list_operations()
    test_interview_problems()