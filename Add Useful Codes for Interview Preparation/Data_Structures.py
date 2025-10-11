"""
Data Structures Python Reference
Author: Mohit Kourav
Description:
This file contains implementations of essential data structures in Python,
including linear structures (Stack, Queue, Deque, Linked Lists), non-linear structures
(Binary Tree, BST, N-ary Tree, Trie), Graph representations (Adjacency List & Matrix),
Hashing (Hash Table / Dictionary), and Heap / Priority Queue. Each structure includes
headings, definitions, explanations, example usage, and time/space complexity.
"""



# --------------------
# DATA STRUCTURES
# --------------------

# --------------------
# LINEAR STRUCTURES
# --------------------

# 1. Stack
# Definition: LIFO structure, push/pop from top.
# Time Complexity: Push/Pop O(1) | Space Complexity: O(n)
# Example Usage: s = Stack(); s.push(1); s.pop()
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        if self.stack:
            return self.stack.pop()
        return None

    def peek(self):
        return self.stack[-1] if self.stack else None

    def is_empty(self):
        return len(self.stack) == 0


# 2. Queue
# Definition: FIFO structure, enqueue at rear, dequeue from front.
# Time Complexity: Enqueue/Dequeue O(1) | Space Complexity: O(n)
# Example Usage: q = Queue(); q.enqueue(1); q.dequeue()
from collections import deque
class Queue:
    def __init__(self):
        self.queue = deque()

    def enqueue(self, val):
        self.queue.append(val)

    def dequeue(self):
        if self.queue:
            return self.queue.popleft()
        return None

    def is_empty(self):
        return len(self.queue) == 0


# 3. Deque
# Definition: Double-ended queue, insert/remove at both ends.
# Time Complexity: O(1) | Space Complexity: O(n)
# Example Usage: d = Deque(); d.append_left(1); d.pop_right()
class Deque:
    def __init__(self):
        self.deque = deque()

    def append_left(self, val):
        self.deque.appendleft(val)

    def append_right(self, val):
        self.deque.append(val)

    def pop_left(self):
        return self.deque.popleft() if self.deque else None

    def pop_right(self):
        return self.deque.pop() if self.deque else None


# 4. Singly Linked List
# Definition: Each node points to the next node. Linear traversal.
# Time Complexity: Search O(n), Insert/Delete at head O(1) | Space Complexity: O(n)
# Example Usage: ll = SinglyLinkedList(); ll.append(1)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        new_node = Node(val)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node


# 5. Doubly Linked List
# Definition: Each node points to next and previous nodes.
# Time Complexity: Search O(n), Insert/Delete at head O(1) | Space Complexity: O(n)
# Example Usage: dll = DoublyLinkedList(); dll.append(1)
class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        new_node = DoublyNode(val)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node
        new_node.prev = curr


# --------------------
# NON-LINEAR STRUCTURES
# --------------------

# 1. Binary Tree
# Definition: Each node has at most 2 children (left and right).
# Time Complexity: Traversal O(n) | Space Complexity: O(n)
# Example Usage: root = BinaryTreeNode(1)
class BinaryTreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


# 2. Binary Search Tree (BST)
# Definition: Binary tree where left < node < right.
# Time Complexity: Search/Insert/Delete O(log n) average | Space Complexity: O(n)
# Example Usage: bst = BST(); bst.insert(5)
class BSTNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = BSTNode(val)
            return
        curr = self.root
        while True:
            if val < curr.val:
                if curr.left:
                    curr = curr.left
                else:
                    curr.left = BSTNode(val)
                    break
            else:
                if curr.right:
                    curr = curr.right
                else:
                    curr.right = BSTNode(val)
                    break


# 3. N-ary Tree
# Definition: Each node can have any number of children.
# Time Complexity: Traversal O(n) | Space Complexity: O(n)
# Example Usage: root = NaryTreeNode(1)
class NaryTreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []


# 4. Trie
# Definition: Tree used for storing strings, each edge represents a character.
# Time Complexity: Insert/Search O(L), L = string length | Space Complexity: O(ALPHABET_SIZE * N * L)
# Example Usage: trie = Trie(); trie.insert('apple')
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end


# --------------------
# GRAPHS
# --------------------

# 1. Graph using Adjacency List
# Definition: Graph represented by dictionary mapping vertices to list of connected vertices.
# Time Complexity: Add Edge O(1), Traversal O(V+E) | Space Complexity: O(V+E)
# Example Usage: g = Graph(3); g.add_edge(0,1); g.add_edge(0,2); g.adj_list
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.adj_list = {i: [] for i in range(vertices)}

    def add_edge(self, u, v):
        self.adj_list[u].append(v)
        # For undirected graph, uncomment below
        # self.adj_list[v].append(u)

    def bfs(self, start):
        visited = [False]*self.V
        queue = deque([start])
        visited[start] = True
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return order

    def dfs(self, start):
        visited = [False]*self.V
        order = []

        def dfs_util(v):
            visited[v] = True
            order.append(v)
            for neighbor in self.adj_list[v]:
                if not visited[neighbor]:
                    dfs_util(neighbor)
        dfs_util(start)
        return order

# Example Usage:
# g = Graph(3)
# g.add_edge(0,1)
# g.add_edge(0,2)
# g.bfs(0)
# g.dfs(0)


# 2. Graph using Adjacency Matrix
# Definition: Graph represented by 2D matrix where matrix[i][j] = 1 if edge exists.
# Time Complexity: Add Edge O(1), Traversal O(V^2) | Space Complexity: O(V^2)
class GraphMatrix:
    def __init__(self, vertices):
        self.V = vertices
        self.matrix = [[0]*vertices for _ in range(vertices)]

    def add_edge(self, u, v):
        self.matrix[u][v] = 1
        # For undirected graph, uncomment below
        # self.matrix[v][u] = 1

    def display(self):
        return self.matrix

# Example Usage:
# gm = GraphMatrix(3)
# gm.add_edge(0,1)
# gm.add_edge(0,2)
# gm.display()


# --------------------
# HASHING
# --------------------

# 1. Hash Table / Dictionary
# Definition: Key-value pairs with average O(1) insert/search using Python dict.
# Time Complexity: O(1) average | Space Complexity: O(n)
class HashTable:
    def __init__(self):
        self.table = {}

    def insert(self, key, value):
        self.table[key] = value

    def search(self, key):
        return self.table.get(key, None)

    def delete(self, key):
        if key in self.table:
            del self.table[key]

# Example Usage:
# ht = HashTable()
# ht.insert('a',1)
# ht.insert('b',2)
# ht.search('a')
# ht.delete('a')


# --------------------
# HEAP / PRIORITY QUEUE
# --------------------

# Definition: Min-Heap or Max-Heap for priority queue operations.
# Time Complexity: Insert/Delete O(log n) | Space Complexity: O(n)
class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        heapq.heappush(self.heap, val)

    def pop(self):
        return heapq.heappop(self.heap) if self.heap else None

    def peek(self):
        return self.heap[0] if self.heap else None

    def heapify_list(self, lst):
        self.heap = lst
        heapq.heapify(self.heap)

# Example Usage:
# mh = MinHeap()
# mh.push(5)
# mh.push(2)
# mh.push(10)
# mh.pop()  # returns 2
# mh.peek() # returns 5
