/**
 * 🚀 JavaScript Data Structures - Modern ES6+ Implementation
 * Author: AI Assistant
 * 
 * Description:
 * This file contains essential data structures implemented in modern JavaScript
 * with ES6+ features, best practices, and performance optimizations.
 */

// ================================
// 1️⃣ ARRAYS - Dynamic Arrays
// ================================

class DynamicArray {
  constructor(initialCapacity = 2) {
    this.capacity = initialCapacity;
    this.length = 0;
    this.data = new Array(this.capacity);
  }

  // O(1) amortized
  push(element) {
    if (this.length >= this.capacity) {
      this._resize();
    }
    this.data[this.length] = element;
    this.length++;
    return this.length;
  }

  // O(1)
  pop() {
    if (this.length === 0) return undefined;
    const element = this.data[this.length - 1];
    this.data[this.length - 1] = undefined;
    this.length--;
    return element;
  }

  // O(1)
  get(index) {
    if (index < 0 || index >= this.length) {
      throw new Error('Index out of bounds');
    }
    return this.data[index];
  }

  // O(1)
  set(index, element) {
    if (index < 0 || index >= this.length) {
      throw new Error('Index out of bounds');
    }
    this.data[index] = element;
  }

  // O(n)
  insert(index, element) {
    if (index < 0 || index > this.length) {
      throw new Error('Index out of bounds');
    }
    
    if (this.length >= this.capacity) {
      this._resize();
    }

    for (let i = this.length; i > index; i--) {
      this.data[i] = this.data[i - 1];
    }
    
    this.data[index] = element;
    this.length++;
  }

  // O(n)
  delete(index) {
    if (index < 0 || index >= this.length) {
      throw new Error('Index out of bounds');
    }

    const element = this.data[index];
    for (let i = index; i < this.length - 1; i++) {
      this.data[i] = this.data[i + 1];
    }
    
    this.data[this.length - 1] = undefined;
    this.length--;
    return element;
  }

  _resize() {
    this.capacity *= 2;
    const newData = new Array(this.capacity);
    for (let i = 0; i < this.length; i++) {
      newData[i] = this.data[i];
    }
    this.data = newData;
  }

  // O(n)
  find(element) {
    for (let i = 0; i < this.length; i++) {
      if (this.data[i] === element) {
        return i;
      }
    }
    return -1;
  }

  // O(n)
  includes(element) {
    return this.find(element) !== -1;
  }

  // O(n)
  reverse() {
    let left = 0;
    let right = this.length - 1;
    
    while (left < right) {
      [this.data[left], this.data[right]] = [this.data[right], this.data[left]];
      left++;
      right--;
    }
  }

  // O(n)
  forEach(callback) {
    for (let i = 0; i < this.length; i++) {
      callback(this.data[i], i, this);
    }
  }

  // O(n)
  map(callback) {
    const result = new DynamicArray();
    for (let i = 0; i < this.length; i++) {
      result.push(callback(this.data[i], i, this));
    }
    return result;
  }

  // O(n)
  filter(callback) {
    const result = new DynamicArray();
    for (let i = 0; i < this.length; i++) {
      if (callback(this.data[i], i, this)) {
        result.push(this.data[i]);
      }
    }
    return result;
  }

  // O(n)
  reduce(callback, initialValue) {
    let accumulator = initialValue;
    for (let i = 0; i < this.length; i++) {
      accumulator = callback(accumulator, this.data[i], i, this);
    }
    return accumulator;
  }

  toString() {
    return `[${this.data.slice(0, this.length).join(', ')}]`;
  }
}

// ================================
// 2️⃣ LINKED LISTS - Singly Linked List
// ================================

class ListNode {
  constructor(value, next = null) {
    this.value = value;
    this.next = next;
  }
}

class LinkedList {
  constructor() {
    this.head = null;
    this.tail = null;
    this.length = 0;
  }

  // O(1)
  prepend(value) {
    const newNode = new ListNode(value, this.head);
    this.head = newNode;
    if (this.length === 0) {
      this.tail = newNode;
    }
    this.length++;
    return this;
  }

  // O(1)
  append(value) {
    const newNode = new ListNode(value);
    if (this.length === 0) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      this.tail.next = newNode;
      this.tail = newNode;
    }
    this.length++;
    return this;
  }

  // O(n)
  insertAt(index, value) {
    if (index < 0 || index > this.length) {
      throw new Error('Index out of bounds');
    }

    if (index === 0) {
      return this.prepend(value);
    }

    if (index === this.length) {
      return this.append(value);
    }

    const newNode = new ListNode(value);
    const prevNode = this.getNodeAt(index - 1);
    newNode.next = prevNode.next;
    prevNode.next = newNode;
    this.length++;
    return this;
  }

  // O(n)
  removeAt(index) {
    if (index < 0 || index >= this.length) {
      throw new Error('Index out of bounds');
    }

    if (index === 0) {
      const value = this.head.value;
      this.head = this.head.next;
      if (this.length === 1) {
        this.tail = null;
      }
      this.length--;
      return value;
    }

    const prevNode = this.getNodeAt(index - 1);
    const nodeToRemove = prevNode.next;
    prevNode.next = nodeToRemove.next;
    
    if (index === this.length - 1) {
      this.tail = prevNode;
    }
    
    this.length--;
    return nodeToRemove.value;
  }

  // O(n)
  get(index) {
    if (index < 0 || index >= this.length) {
      throw new Error('Index out of bounds');
    }
    return this.getNodeAt(index).value;
  }

  // O(n)
  set(index, value) {
    if (index < 0 || index >= this.length) {
      throw new Error('Index out of bounds');
    }
    const node = this.getNodeAt(index);
    node.value = value;
  }

  getNodeAt(index) {
    let current = this.head;
    for (let i = 0; i < index; i++) {
      current = current.next;
    }
    return current;
  }

  // O(n)
  find(value) {
    let current = this.head;
    let index = 0;
    
    while (current) {
      if (current.value === value) {
        return index;
      }
      current = current.next;
      index++;
    }
    
    return -1;
  }

  // O(n)
  includes(value) {
    return this.find(value) !== -1;
  }

  // O(n)
  reverse() {
    let prev = null;
    let current = this.head;
    this.tail = this.head;
    
    while (current) {
      const next = current.next;
      current.next = prev;
      prev = current;
      current = next;
    }
    
    this.head = prev;
    return this;
  }

  // O(n)
  forEach(callback) {
    let current = this.head;
    let index = 0;
    
    while (current) {
      callback(current.value, index, this);
      current = current.next;
      index++;
    }
  }

  // O(n)
  map(callback) {
    const result = new LinkedList();
    let current = this.head;
    let index = 0;
    
    while (current) {
      result.append(callback(current.value, index, this));
      current = current.next;
      index++;
    }
    
    return result;
  }

  // O(n)
  filter(callback) {
    const result = new LinkedList();
    let current = this.head;
    let index = 0;
    
    while (current) {
      if (callback(current.value, index, this)) {
        result.append(current.value);
      }
      current = current.next;
      index++;
    }
    
    return result;
  }

  // O(n)
  reduce(callback, initialValue) {
    let accumulator = initialValue;
    let current = this.head;
    let index = 0;
    
    while (current) {
      accumulator = callback(accumulator, current.value, index, this);
      current = current.next;
      index++;
    }
    
    return accumulator;
  }

  toArray() {
    const result = [];
    let current = this.head;
    
    while (current) {
      result.push(current.value);
      current = current.next;
    }
    
    return result;
  }

  toString() {
    return this.toArray().join(' -> ');
  }
}

// ================================
// 3️⃣ STACKS - LIFO Data Structure
// ================================

class Stack {
  constructor() {
    this.items = [];
  }

  // O(1)
  push(element) {
    this.items.push(element);
    return this;
  }

  // O(1)
  pop() {
    if (this.isEmpty()) {
      throw new Error('Stack is empty');
    }
    return this.items.pop();
  }

  // O(1)
  peek() {
    if (this.isEmpty()) {
      throw new Error('Stack is empty');
    }
    return this.items[this.items.length - 1];
  }

  // O(1)
  isEmpty() {
    return this.items.length === 0;
  }

  // O(1)
  size() {
    return this.items.length;
  }

  // O(n)
  clear() {
    this.items = [];
  }

  // O(n)
  forEach(callback) {
    this.items.forEach(callback);
  }

  // O(n)
  map(callback) {
    return this.items.map(callback);
  }

  // O(n)
  filter(callback) {
    return this.items.filter(callback);
  }

  toString() {
    return `[${this.items.join(', ')}]`;
  }
}

// ================================
// 4️⃣ QUEUES - FIFO Data Structure
// ================================

class Queue {
  constructor() {
    this.items = [];
  }

  // O(1)
  enqueue(element) {
    this.items.push(element);
    return this;
  }

  // O(n) - can be optimized with circular buffer
  dequeue() {
    if (this.isEmpty()) {
      throw new Error('Queue is empty');
    }
    return this.items.shift();
  }

  // O(1)
  front() {
    if (this.isEmpty()) {
      throw new Error('Queue is empty');
    }
    return this.items[0];
  }

  // O(1)
  rear() {
    if (this.isEmpty()) {
      throw new Error('Queue is empty');
    }
    return this.items[this.items.length - 1];
  }

  // O(1)
  isEmpty() {
    return this.items.length === 0;
  }

  // O(1)
  size() {
    return this.items.length;
  }

  // O(n)
  clear() {
    this.items = [];
  }

  // O(n)
  forEach(callback) {
    this.items.forEach(callback);
  }

  toString() {
    return `[${this.items.join(', ')}]`;
  }
}

// ================================
// 5️⃣ HASH TABLES - Key-Value Storage
// ================================

class HashTable {
  constructor(size = 16) {
    this.size = size;
    this.buckets = new Array(size).fill(null).map(() => []);
    this.count = 0;
  }

  _hash(key) {
    let hash = 0;
    const keyStr = String(key);
    
    for (let i = 0; i < keyStr.length; i++) {
      const char = keyStr.charCodeAt(i);
      hash = ((hash << 5) - hash + char) & 0x7fffffff;
    }
    
    return hash % this.size;
  }

  // O(1) average, O(n) worst case
  set(key, value) {
    const index = this._hash(key);
    const bucket = this.buckets[index];
    
    // Check if key already exists
    for (let i = 0; i < bucket.length; i++) {
      if (bucket[i][0] === key) {
        bucket[i][1] = value;
        return this;
      }
    }
    
    // Add new key-value pair
    bucket.push([key, value]);
    this.count++;
    
    // Resize if load factor is too high
    if (this.count > this.size * 0.75) {
      this._resize();
    }
    
    return this;
  }

  // O(1) average, O(n) worst case
  get(key) {
    const index = this._hash(key);
    const bucket = this.buckets[index];
    
    for (let i = 0; i < bucket.length; i++) {
      if (bucket[i][0] === key) {
        return bucket[i][1];
      }
    }
    
    return undefined;
  }

  // O(1) average, O(n) worst case
  delete(key) {
    const index = this._hash(key);
    const bucket = this.buckets[index];
    
    for (let i = 0; i < bucket.length; i++) {
      if (bucket[i][0] === key) {
        bucket.splice(i, 1);
        this.count--;
        return true;
      }
    }
    
    return false;
  }

  // O(1)
  has(key) {
    return this.get(key) !== undefined;
  }

  // O(n)
  keys() {
    const keys = [];
    for (const bucket of this.buckets) {
      for (const [key] of bucket) {
        keys.push(key);
      }
    }
    return keys;
  }

  // O(n)
  values() {
    const values = [];
    for (const bucket of this.buckets) {
      for (const [, value] of bucket) {
        values.push(value);
      }
    }
    return values;
  }

  // O(n)
  entries() {
    const entries = [];
    for (const bucket of this.buckets) {
      for (const [key, value] of bucket) {
        entries.push([key, value]);
      }
    }
    return entries;
  }

  // O(n)
  forEach(callback) {
    for (const [key, value] of this.entries()) {
      callback(value, key, this);
    }
  }

  // O(n)
  _resize() {
    const oldBuckets = this.buckets;
    this.size *= 2;
    this.buckets = new Array(this.size).fill(null).map(() => []);
    this.count = 0;
    
    for (const bucket of oldBuckets) {
      for (const [key, value] of bucket) {
        this.set(key, value);
      }
    }
  }

  // O(1)
  get loadFactor() {
    return this.count / this.size;
  }

  toString() {
    return `{${this.entries().map(([k, v]) => `${k}: ${v}`).join(', ')}}`;
  }
}

// ================================
// 6️⃣ BINARY TREES - Tree Data Structure
// ================================

class TreeNode {
  constructor(value, left = null, right = null) {
    this.value = value;
    this.left = left;
    this.right = right;
  }
}

class BinaryTree {
  constructor() {
    this.root = null;
  }

  // O(log n) average, O(n) worst case
  insert(value) {
    const newNode = new TreeNode(value);
    
    if (!this.root) {
      this.root = newNode;
      return this;
    }
    
    let current = this.root;
    while (true) {
      if (value < current.value) {
        if (!current.left) {
          current.left = newNode;
          break;
        }
        current = current.left;
      } else {
        if (!current.right) {
          current.right = newNode;
          break;
        }
        current = current.right;
      }
    }
    
    return this;
  }

  // O(log n) average, O(n) worst case
  find(value) {
    let current = this.root;
    
    while (current) {
      if (value === current.value) {
        return current;
      } else if (value < current.value) {
        current = current.left;
      } else {
        current = current.right;
      }
    }
    
    return null;
  }

  // O(log n) average, O(n) worst case
  contains(value) {
    return this.find(value) !== null;
  }

  // O(n)
  inOrderTraversal(callback) {
    const traverse = (node) => {
      if (node) {
        traverse(node.left);
        callback(node.value);
        traverse(node.right);
      }
    };
    traverse(this.root);
  }

  // O(n)
  preOrderTraversal(callback) {
    const traverse = (node) => {
      if (node) {
        callback(node.value);
        traverse(node.left);
        traverse(node.right);
      }
    };
    traverse(this.root);
  }

  // O(n)
  postOrderTraversal(callback) {
    const traverse = (node) => {
      if (node) {
        traverse(node.left);
        traverse(node.right);
        callback(node.value);
      }
    };
    traverse(this.root);
  }

  // O(n)
  levelOrderTraversal(callback) {
    if (!this.root) return;
    
    const queue = [this.root];
    
    while (queue.length > 0) {
      const node = queue.shift();
      callback(node.value);
      
      if (node.left) queue.push(node.left);
      if (node.right) queue.push(node.right);
    }
  }

  // O(n)
  getHeight() {
    const getHeight = (node) => {
      if (!node) return 0;
      return 1 + Math.max(getHeight(node.left), getHeight(node.right));
    };
    return getHeight(this.root);
  }

  // O(n)
  getSize() {
    let size = 0;
    this.inOrderTraversal(() => size++);
    return size;
  }

  // O(n)
  toArray() {
    const result = [];
    this.inOrderTraversal(value => result.push(value));
    return result;
  }

  toString() {
    return this.toArray().join(', ');
  }
}

// ================================
// 7️⃣ GRAPHS - Adjacency List Representation
// ================================

class Graph {
  constructor() {
    this.adjacencyList = new Map();
  }

  // O(1)
  addVertex(vertex) {
    if (!this.adjacencyList.has(vertex)) {
      this.adjacencyList.set(vertex, []);
    }
    return this;
  }

  // O(1)
  addEdge(vertex1, vertex2) {
    this.addVertex(vertex1);
    this.addVertex(vertex2);
    
    this.adjacencyList.get(vertex1).push(vertex2);
    this.adjacencyList.get(vertex2).push(vertex1);
    
    return this;
  }

  // O(1)
  addDirectedEdge(from, to) {
    this.addVertex(from);
    this.addVertex(to);
    
    this.adjacencyList.get(from).push(to);
    return this;
  }

  // O(1)
  removeEdge(vertex1, vertex2) {
    if (this.adjacencyList.has(vertex1)) {
      const index1 = this.adjacencyList.get(vertex1).indexOf(vertex2);
      if (index1 > -1) {
        this.adjacencyList.get(vertex1).splice(index1, 1);
      }
    }
    
    if (this.adjacencyList.has(vertex2)) {
      const index2 = this.adjacencyList.get(vertex2).indexOf(vertex1);
      if (index2 > -1) {
        this.adjacencyList.get(vertex2).splice(index2, 1);
      }
    }
    
    return this;
  }

  // O(n)
  removeVertex(vertex) {
    if (this.adjacencyList.has(vertex)) {
      const neighbors = this.adjacencyList.get(vertex);
      for (const neighbor of neighbors) {
        this.removeEdge(vertex, neighbor);
      }
      this.adjacencyList.delete(vertex);
    }
    return this;
  }

  // O(1)
  getNeighbors(vertex) {
    return this.adjacencyList.get(vertex) || [];
  }

  // O(1)
  hasVertex(vertex) {
    return this.adjacencyList.has(vertex);
  }

  // O(1)
  hasEdge(vertex1, vertex2) {
    return this.adjacencyList.has(vertex1) && 
           this.adjacencyList.get(vertex1).includes(vertex2);
  }

  // O(n)
  getVertices() {
    return Array.from(this.adjacencyList.keys());
  }

  // O(n)
  getEdges() {
    const edges = [];
    for (const [vertex, neighbors] of this.adjacencyList) {
      for (const neighbor of neighbors) {
        edges.push([vertex, neighbor]);
      }
    }
    return edges;
  }

  // O(n)
  getSize() {
    return this.adjacencyList.size;
  }

  // O(n)
  clear() {
    this.adjacencyList.clear();
  }

  toString() {
    const result = [];
    for (const [vertex, neighbors] of this.adjacencyList) {
      result.push(`${vertex}: [${neighbors.join(', ')}]`);
    }
    return result.join('\n');
  }
}

// ================================
// 8️⃣ EXPORTS - Module Exports
// ================================

export {
  DynamicArray,
  LinkedList,
  ListNode,
  Stack,
  Queue,
  HashTable,
  BinaryTree,
  TreeNode,
  Graph
};

// ================================
// 9️⃣ DEMONSTRATION - Usage Examples
// ================================

if (typeof window === 'undefined') {
  // Node.js environment
  console.log('=== JavaScript Data Structures Demo ===\n');
  
  // Dynamic Array
  console.log('1. Dynamic Array:');
  const arr = new DynamicArray();
  arr.push(1).push(2).push(3);
  console.log(`Array: ${arr}`);
  console.log(`Length: ${arr.length}`);
  console.log(`Capacity: ${arr.capacity}\n`);
  
  // Linked List
  console.log('2. Linked List:');
  const list = new LinkedList();
  list.append(1).append(2).append(3);
  console.log(`List: ${list}`);
  console.log(`Length: ${list.length}\n`);
  
  // Stack
  console.log('3. Stack:');
  const stack = new Stack();
  stack.push(1).push(2).push(3);
  console.log(`Stack: ${stack}`);
  console.log(`Peek: ${stack.peek()}`);
  console.log(`Pop: ${stack.pop()}\n`);
  
  // Queue
  console.log('4. Queue:');
  const queue = new Queue();
  queue.enqueue(1).enqueue(2).enqueue(3);
  console.log(`Queue: ${queue}`);
  console.log(`Front: ${queue.front()}`);
  console.log(`Dequeue: ${queue.dequeue()}\n`);
  
  // Hash Table
  console.log('5. Hash Table:');
  const hashTable = new HashTable();
  hashTable.set('name', 'John').set('age', 30).set('city', 'New York');
  console.log(`Hash Table: ${hashTable}`);
  console.log(`Get 'name': ${hashTable.get('name')}`);
  console.log(`Load Factor: ${hashTable.loadFactor}\n`);
  
  // Binary Tree
  console.log('6. Binary Tree:');
  const tree = new BinaryTree();
  tree.insert(5).insert(3).insert(7).insert(1).insert(4);
  console.log(`Tree (in-order): ${tree}`);
  console.log(`Height: ${tree.getHeight()}`);
  console.log(`Size: ${tree.getSize()}\n`);
  
  // Graph
  console.log('7. Graph:');
  const graph = new Graph();
  graph.addEdge('A', 'B').addEdge('B', 'C').addEdge('C', 'D');
  console.log(`Graph:\n${graph}`);
  console.log(`Vertices: ${graph.getVertices()}`);
  console.log(`Edges: ${graph.getEdges().length}`);
}
