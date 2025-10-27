"""
Comprehensive Stack and Queue Implementation with Interview Problems

This module implements various stack and queue data structures with common
interview problems. Essential for technical interviews at top tech companies.

Includes: Stack, Queue, Deque, Priority Queue, and Monotonic Stack/Queue
Time/Space complexity analysis included for each operation.
"""

from collections import deque
import heapq
from typing import List, Optional


class Stack:
    """
    Stack implementation using list.
    LIFO (Last In, First Out) principle.
    """
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top. Time: O(1), Space: O(1)"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item. Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing. Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty. Time: O(1), Space: O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return stack size. Time: O(1), Space: O(1)"""
        return len(self.items)
    
    def display(self):
        """Display stack contents."""
        return f"Stack: {self.items} (top -> bottom)"


class Queue:
    """
    Queue implementation using deque for efficiency.
    FIFO (First In, First Out) principle.
    """
    
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        """Add item to rear. Time: O(1), Space: O(1)"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item. Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()
    
    def front(self):
        """Return front item without removing. Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def rear(self):
        """Return rear item without removing. Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[-1]
    
    def is_empty(self):
        """Check if queue is empty. Time: O(1), Space: O(1)"""
        return len(self.items) == 0
    
    def size(self):
        """Return queue size. Time: O(1), Space: O(1)"""
        return len(self.items)
    
    def display(self):
        """Display queue contents."""
        return f"Queue: {list(self.items)} (front -> rear)"


class CircularQueue:
    """
    Circular Queue implementation with fixed size.
    Space-efficient queue with wrap-around behavior.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0
        self.rear = -1
        self.count = 0
    
    def enqueue(self, item):
        """Add item to rear. Time: O(1), Space: O(1)"""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.count += 1
    
    def dequeue(self):
        """Remove and return front item. Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        
        item = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.count -= 1
        return item
    
    def is_empty(self):
        """Check if queue is empty. Time: O(1), Space: O(1)"""
        return self.count == 0
    
    def is_full(self):
        """Check if queue is full. Time: O(1), Space: O(1)"""
        return self.count == self.capacity
    
    def size(self):
        """Return current size. Time: O(1), Space: O(1)"""
        return self.count


class PriorityQueue:
    """
    Priority Queue implementation using heap.
    Elements with higher priority are served first.
    """
    
    def __init__(self, max_heap=False):
        self.heap = []
        self.max_heap = max_heap
    
    def enqueue(self, item, priority):
        """Add item with priority. Time: O(log n), Space: O(1)"""
        if self.max_heap:
            heapq.heappush(self.heap, (-priority, item))
        else:
            heapq.heappush(self.heap, (priority, item))
    
    def dequeue(self):
        """Remove highest priority item. Time: O(log n), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, item = heapq.heappop(self.heap)
        if self.max_heap:
            priority = -priority
        return item, priority
    
    def peek(self):
        """Return highest priority item. Time: O(1), Space: O(1)"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        priority, item = self.heap[0]
        if self.max_heap:
            priority = -priority
        return item, priority
    
    def is_empty(self):
        """Check if empty. Time: O(1), Space: O(1)"""
        return len(self.heap) == 0
    
    def size(self):
        """Return size. Time: O(1), Space: O(1)"""
        return len(self.heap)


class MonotonicStack:
    """
    Monotonic Stack for finding next/previous greater/smaller elements.
    Used in many interview problems.
    """
    
    def __init__(self, increasing=True):
        self.stack = []
        self.increasing = increasing  # True for increasing, False for decreasing
    
    def push(self, item):
        """Push item while maintaining monotonic property."""
        if self.increasing:
            while self.stack and self.stack[-1] > item:
                self.stack.pop()
        else:
            while self.stack and self.stack[-1] < item:
                self.stack.pop()
        
        self.stack.append(item)
    
    def get_stack(self):
        """Return current stack."""
        return self.stack.copy()


class MinStack:
    """
    Stack that supports push, pop, top, and retrieving minimum in O(1).
    LeetCode 155 - Min Stack
    """
    
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        """Push value. Time: O(1), Space: O(1)"""
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        """Pop value. Time: O(1), Space: O(1)"""
        if not self.stack:
            raise IndexError("Stack is empty")
        
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val
    
    def top(self):
        """Get top value. Time: O(1), Space: O(1)"""
        if not self.stack:
            raise IndexError("Stack is empty")
        return self.stack[-1]
    
    def get_min(self):
        """Get minimum value. Time: O(1), Space: O(1)"""
        if not self.min_stack:
            raise IndexError("Stack is empty")
        return self.min_stack[-1]


# Interview Problem Functions

def valid_parentheses(s: str) -> bool:
    """
    Check if parentheses are valid.
    LeetCode 20 - Valid Parentheses
    Time: O(n), Space: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return not stack


def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Find days until warmer temperature.
    LeetCode 739 - Daily Temperatures
    Time: O(n), Space: O(n)
    """
    result = [0] * len(temperatures)
    stack = []  # Stack of indices
    
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    
    return result


def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find next greater element for each element in nums1.
    LeetCode 496 - Next Greater Element I
    Time: O(n + m), Space: O(n)
    """
    stack = []
    next_greater = {}
    
    # Find next greater for all elements in nums2
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # Build result for nums1
    return [next_greater.get(num, -1) for num in nums1]


def largest_rectangle_histogram(heights: List[int]) -> int:
    """
    Find largest rectangle in histogram.
    LeetCode 84 - Largest Rectangle in Histogram
    Time: O(n), Space: O(n)
    """
    stack = []
    max_area = 0
    heights.append(0)  # Add sentinel
    
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    
    return max_area


def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    """
    Find maximum in each sliding window.
    LeetCode 239 - Sliding Window Maximum
    Time: O(n), Space: O(k)
    """
    from collections import deque
    
    dq = deque()  # Store indices
    result = []
    
    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements from rear
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # Add to result when window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result


def implement_queue_using_stacks():
    """
    Implement Queue using two stacks.
    LeetCode 232 - Implement Queue using Stacks
    """
    class MyQueue:
        def __init__(self):
            self.input_stack = []
            self.output_stack = []
        
        def push(self, x):
            """Time: O(1)"""
            self.input_stack.append(x)
        
        def pop(self):
            """Time: O(1) amortized"""
            self._move_to_output()
            return self.output_stack.pop()
        
        def peek(self):
            """Time: O(1) amortized"""
            self._move_to_output()
            return self.output_stack[-1]
        
        def empty(self):
            """Time: O(1)"""
            return not self.input_stack and not self.output_stack
        
        def _move_to_output(self):
            if not self.output_stack:
                while self.input_stack:
                    self.output_stack.append(self.input_stack.pop())
    
    return MyQueue


def test_basic_structures():
    """Test basic stack and queue operations."""
    print("=== Testing Stack ===")
    stack = Stack()
    
    for i in range(1, 6):
        stack.push(i)
    print(f"After pushing 1-5: {stack.display()}")
    
    print(f"Peek: {stack.peek()}")
    print(f"Pop: {stack.pop()}")
    print(f"After pop: {stack.display()}")
    
    print("\n=== Testing Queue ===")
    queue = Queue()
    
    for i in range(1, 6):
        queue.enqueue(i)
    print(f"After enqueueing 1-5: {queue.display()}")
    
    print(f"Front: {queue.front()}")
    print(f"Dequeue: {queue.dequeue()}")
    print(f"After dequeue: {queue.display()}")
    
    print("\n=== Testing Priority Queue ===")
    pq = PriorityQueue()
    
    tasks = [("Low priority", 1), ("High priority", 5), ("Medium priority", 3)]
    for task, priority in tasks:
        pq.enqueue(task, priority)
    
    print("Processing tasks by priority:")
    while not pq.is_empty():
        task, priority = pq.dequeue()
        print(f"  {task} (priority: {priority})")


def test_interview_problems():
    """Test interview-specific problems."""
    print("\n=== Testing Interview Problems ===")
    
    # Test valid parentheses
    test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}"]
    print("Valid Parentheses:")
    for case in test_cases:
        result = valid_parentheses(case)
        print(f"  '{case}' -> {result}")
    
    # Test daily temperatures
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    result = daily_temperatures(temps)
    print(f"\nDaily Temperatures:")
    print(f"  Input: {temps}")
    print(f"  Output: {result}")
    
    # Test next greater element
    nums1 = [4, 1, 2]
    nums2 = [1, 3, 4, 2]
    result = next_greater_element(nums1, nums2)
    print(f"\nNext Greater Element:")
    print(f"  nums1: {nums1}, nums2: {nums2}")
    print(f"  Result: {result}")
    
    # Test sliding window maximum
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    result = sliding_window_maximum(nums, k)
    print(f"\nSliding Window Maximum (k={k}):")
    print(f"  Input: {nums}")
    print(f"  Output: {result}")
    
    # Test MinStack
    print(f"\nTesting MinStack:")
    min_stack = MinStack()
    operations = [
        ("push", -2), ("push", 0), ("push", -3),
        ("get_min", None), ("pop", None), ("top", None), ("get_min", None)
    ]
    
    for op, val in operations:
        if op == "push":
            min_stack.push(val)
            print(f"  push({val})")
        elif op == "pop":
            result = min_stack.pop()
            print(f"  pop() -> {result}")
        elif op == "top":
            result = min_stack.top()
            print(f"  top() -> {result}")
        elif op == "get_min":
            result = min_stack.get_min()
            print(f"  get_min() -> {result}")


if __name__ == "__main__":
    test_basic_structures()
    test_interview_problems()