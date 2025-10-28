# 🚀 JavaScript Algorithms - Modern ES6+ Implementation

This folder contains comprehensive solutions to common algorithm problems implemented in modern JavaScript (ES6+) with best practices and performance optimizations.

## 🎯 What's Included

### Data Structures
- **Arrays** - Dynamic arrays, 2D arrays, matrix operations
- **Linked Lists** - Singly, doubly, circular linked lists
- **Stacks & Queues** - LIFO/FIFO data structures
- **Trees** - Binary trees, BST, AVL, Trie
- **Graphs** - Adjacency list/matrix, BFS/DFS
- **Hash Tables** - Maps, Sets, custom implementations
- **Heaps** - Min/Max heaps, priority queues

### Algorithms
- **Sorting** - Quick sort, merge sort, heap sort, radix sort
- **Searching** - Binary search, linear search, interpolation search
- **Dynamic Programming** - Memoization, tabulation, optimization
- **Greedy Algorithms** - Activity selection, Huffman coding
- **Graph Algorithms** - Shortest path, MST, topological sort
- **String Algorithms** - KMP, Rabin-Karp, pattern matching

### Modern JavaScript Features
- **ES6+ Syntax** - Arrow functions, destructuring, spread operator
- **Async/Await** - Promise handling, async operations
- **Generators** - Iterator protocols, lazy evaluation
- **Modules** - ES6 modules, import/export
- **Classes** - OOP patterns, inheritance, encapsulation
- **Functional Programming** - Pure functions, immutability

## 📁 Folder Structure

```
JavaScript-Algorithms/
├── README.md
├── data-structures/
│   ├── arrays.js
│   ├── linked-lists.js
│   ├── stacks-queues.js
│   ├── trees.js
│   ├── graphs.js
│   └── hash-tables.js
├── algorithms/
│   ├── sorting.js
│   ├── searching.js
│   ├── dynamic-programming.js
│   ├── greedy.js
│   └── graph-algorithms.js
├── interview-questions/
│   ├── arrays.js
│   ├── strings.js
│   ├── trees.js
│   ├── graphs.js
│   └── system-design.js
├── utilities/
│   ├── helpers.js
│   ├── test-utils.js
│   └── performance.js
└── examples/
    ├── real-world-examples.js
    └── leetcode-solutions.js
```

## 🛠️ Modern JavaScript Features Used

### ES6+ Syntax
```javascript
// Arrow functions
const add = (a, b) => a + b;

// Destructuring
const { name, age } = person;

// Spread operator
const newArray = [...oldArray, newItem];

// Template literals
const message = `Hello ${name}!`;

// Default parameters
const greet = (name = 'World') => `Hello ${name}!`;
```

### Classes and Modules
```javascript
// ES6 Classes
class LinkedList {
  constructor() {
    this.head = null;
  }
  
  add(value) {
    // Implementation
  }
}

// ES6 Modules
export class BinaryTree {
  // Implementation
}

import { BinaryTree } from './trees.js';
```

### Async/Await
```javascript
// Async operations
async function fetchData() {
  try {
    const response = await fetch('/api/data');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Generators
```javascript
// Generator functions
function* fibonacci() {
  let a = 0, b = 1;
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}
```

## ⏱️ Performance Considerations

### Time Complexity
- **O(1)** - Constant time operations
- **O(log n)** - Logarithmic operations (binary search)
- **O(n)** - Linear operations (linear search)
- **O(n log n)** - Linearithmic operations (merge sort)
- **O(n²)** - Quadratic operations (bubble sort)
- **O(2ⁿ)** - Exponential operations (recursive Fibonacci)

### Space Complexity
- **O(1)** - Constant space
- **O(n)** - Linear space
- **O(n²)** - Quadratic space

### Optimization Techniques
- **Memoization** - Caching computed results
- **Tabulation** - Bottom-up dynamic programming
- **Tail Recursion** - Optimizing recursive calls
- **Lazy Evaluation** - Computing values on demand

## 🚀 How to Run

### Node.js
```bash
node data-structures/arrays.js
```

### Browser
```html
<script type="module" src="algorithms/sorting.js"></script>
```

### Testing
```bash
npm test
# or
node test-utils.js
```

## 📖 Learning Resources

- [MDN JavaScript Guide](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
- [LeetCode JavaScript Solutions](https://leetcode.com/tag/javascript/)
- [HackerRank JavaScript Challenges](https://www.hackerrank.com/domains/javascript)
- [JavaScript Algorithms Repository](https://github.com/trekhleb/javascript-algorithms)

## 🤝 Contributing

Feel free to add more problems, improve existing solutions, or add implementations with different approaches!

---

**Happy Coding! 🎉**
