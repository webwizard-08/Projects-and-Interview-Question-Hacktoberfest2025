# 🗂️ Hash Table/HashMap - Important Interview Problems

This folder contains comprehensive solutions to common hash table-based interview problems across multiple programming languages.

## 🎯 What is a Hash Table?

A Hash Table (also called HashMap or Dictionary) is a data structure that implements an associative array abstract data type, a structure that can map keys to values. It uses a hash function to compute an index into an array of buckets or slots.

## 📁 Problems Covered

### Basic Hash Table Operations
1. **Implement Hash Table** - Basic hash table with collision handling
2. **Two Sum** - Find two numbers that add up to target
3. **Valid Anagram** - Check if two strings are anagrams
4. **Group Anagrams** - Group strings that are anagrams
5. **First Non-Repeating Character** - Find first unique character

### Frequency and Counting Problems
6. **Majority Element** - Find element appearing more than n/2 times
7. **Top K Frequent Elements** - Find k most frequent elements
8. **Sort Characters by Frequency** - Sort string by character frequency
9. **Find All Anagrams in String** - Find all anagram substrings
10. **Longest Substring Without Repeating Characters** - Sliding window + hash

### Advanced Hash Table Problems
11. **LRU Cache** - Least Recently Used cache implementation
12. **LFU Cache** - Least Frequently Used cache implementation
13. **Design HashMap** - Custom hash map implementation
14. **Design HashSet** - Custom hash set implementation
15. **Insert Delete GetRandom O(1)** - Randomized set with O(1) operations

### String and Pattern Matching
16. **Word Pattern** - Check if string follows given pattern
17. **Isomorphic Strings** - Check if two strings are isomorphic
18. **Minimum Window Substring** - Find minimum window containing all characters
19. **Longest Substring with At Most K Distinct Characters** - Sliding window
20. **Substring with Concatenation of All Words** - Complex pattern matching

### Array and Matrix Problems
21. **Contains Duplicate** - Check if array has duplicates
22. **Contains Duplicate II** - Check for duplicates within k distance
23. **Contains Duplicate III** - Check for duplicates within k distance and t value
24. **Intersection of Two Arrays** - Find common elements
25. **Intersection of Two Arrays II** - Find common elements with frequency

### Design Problems
26. **Design Twitter** - Social media feed system
27. **Design Underground System** - Transportation tracking
28. **Design Search Autocomplete** - Trie + frequency tracking
29. **Design Rate Limiter** - Token bucket algorithm
30. **Design Hit Counter** - Rate limiting with time windows

## 🛠️ Implementation Languages

- **Java** - HashMap, HashSet, ConcurrentHashMap
- **Python** - dict, defaultdict, Counter, OrderedDict
- **C++** - unordered_map, unordered_set, map, set
- **JavaScript** - Map, Set, Object

## ⏱️ Time Complexities

| Operation | Average Case | Worst Case |
|-----------|--------------|------------|
| Insert    | O(1)         | O(n)       |
| Delete    | O(1)         | O(n)       |
| Search    | O(1)         | O(n)       |
| Update    | O(1)         | O(n)       |

## 🔧 Hash Function Properties

1. **Deterministic** - Same input always produces same output
2. **Uniform Distribution** - Outputs should be evenly distributed
3. **Fast Computation** - Should be O(1) time complexity
4. **Minimal Collisions** - Should minimize hash collisions

## 🚀 How to Run

### Java
```bash
javac HashTableProblems.java
java HashTableProblems
```

### Python
```bash
python hash_table_problems.py
```

### C++
```bash
g++ -o hash_table_problems *.cpp
./hash_table_problems
```

### JavaScript
```bash
node hash_table_problems.js
```

## 📖 Learning Resources

- [GeeksforGeeks - Hash Table](https://www.geeksforgeeks.org/hashing-data-structure/)
- [LeetCode Hash Table Problems](https://leetcode.com/tag/hash-table/)
- [HackerRank Hash Challenges](https://www.hackerrank.com/domains/data-structures/hash)

## 🤝 Contributing

Feel free to add more problems, improve existing solutions, or add implementations in other languages!

---

**Happy Coding! 🎉**
