# 🎯 C++ Interview Tips & Best Practices

## 📋 Quick Reference Guide

### 🏗️ Object-Oriented Programming Checklist

**Always mention these key concepts:**
- ✅ **Encapsulation**: Data hiding with private/protected members
- ✅ **Inheritance**: Code reusability through base/derived classes
- ✅ **Polymorphism**: Runtime binding with virtual functions
- ✅ **Abstraction**: Interface definition with pure virtual functions

**Common OOP Interview Questions:**
1. "Explain the four pillars of OOP with examples"
2. "What's the difference between virtual and pure virtual functions?"
3. "When would you use inheritance vs composition?"
4. "Explain the difference between public, private, and protected inheritance"

### 📚 STL Mastery Checklist

**Essential STL Knowledge:**
- ✅ **Containers**: vector, list, map, set, unordered_map
- ✅ **Algorithms**: sort, find, transform, accumulate
- ✅ **Iterators**: begin(), end(), range-based loops
- ✅ **Templates**: Function and class templates

**STL Interview Questions:**
1. "When would you use std::vector vs std::list?"
2. "What's the time complexity of std::map operations?"
3. "Explain the difference between std::map and std::unordered_map"
4. "How do you sort a vector of custom objects?"

### 🧠 Memory Management Checklist

**Critical Memory Concepts:**
- ✅ **RAII**: Resource Acquisition Is Initialization
- ✅ **Smart Pointers**: unique_ptr, shared_ptr, weak_ptr
- ✅ **Rule of Three/Five**: Copy constructor, assignment, destructor
- ✅ **Exception Safety**: No memory leaks even with exceptions

**Memory Management Questions:**
1. "What's the difference between stack and heap memory?"
2. "When would you use unique_ptr vs shared_ptr?"
3. "How do you prevent memory leaks in C++?"
4. "Explain RAII with an example"

## 🚀 Coding Interview Strategies

### 1. **Start with the Problem Understanding**
- Ask clarifying questions
- Identify input/output requirements
- Consider edge cases

### 2. **Choose the Right Data Structure**
```cpp
// For frequent insertions/deletions at both ends
std::deque<int> dq;

// For key-value lookups (ordered)
std::map<std::string, int> m;

// For key-value lookups (unordered, faster)
std::unordered_map<std::string, int> um;

// For dynamic arrays
std::vector<int> vec;
```

### 3. **Use Modern C++ Features**
```cpp
// Prefer auto for type deduction
auto result = std::find(vec.begin(), vec.end(), target);

// Use range-based loops
for (const auto& item : container) {
    // process item
}

// Use smart pointers
std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();

// Use lambdas
std::sort(vec.begin(), vec.end(), [](int a, int b) { return a > b; });
```

### 4. **Handle Edge Cases**
- Empty containers
- Null pointers
- Invalid input
- Memory constraints

## 💡 Common Interview Patterns

### Pattern 1: Two Pointers
```cpp
// Find pair with given sum in sorted array
bool hasPair(std::vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) return true;
        else if (sum < target) left++;
        else right--;
    }
    return false;
}
```

### Pattern 2: Sliding Window
```cpp
// Maximum sum of subarray of size k
int maxSumSubarray(std::vector<int>& arr, int k) {
    int windowSum = 0, maxSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    maxSum = windowSum;
    
    for (int i = k; i < arr.size(); i++) {
        windowSum += arr[i] - arr[i - k];
        maxSum = std::max(maxSum, windowSum);
    }
    return maxSum;
}
```

### Pattern 3: Hash Map
```cpp
// Two Sum problem
std::vector<int> twoSum(std::vector<int>& nums, int target) {
    std::unordered_map<int, int> map;
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (map.find(complement) != map.end()) {
            return {map[complement], i};
        }
        map[nums[i]] = i;
    }
    return {};
}
```

## 🎯 Behavioral Questions

### Technical Decision Making
- "Describe a time when you had to choose between different algorithms"
- "How do you approach debugging memory issues?"
- "Explain a complex C++ project you worked on"

### Problem Solving
- "How would you design a cache system?"
- "What's your approach to optimizing C++ code?"
- "How do you ensure code quality in C++ projects?"

## 📚 Study Resources

### Books
- **Effective C++** by Scott Meyers
- **Modern C++** by Scott Meyers
- **C++ Primer** by Lippman, Lajoie, and Moo

### Online Resources
- **cppreference.com** - Official C++ reference
- **Stack Overflow** - Community Q&A
- **LeetCode** - Coding practice

### Practice Platforms
- **LeetCode** - Algorithm problems
- **HackerRank** - C++ challenges
- **CodeSignal** - Technical assessments

## ⚠️ Common Mistakes to Avoid

1. **Memory Leaks**: Always use smart pointers or RAII
2. **Buffer Overflows**: Validate array bounds
3. **Dangling Pointers**: Use weak_ptr for non-owning references
4. **Exception Unsafety**: Ensure RAII in all code paths
5. **Inefficient STL Usage**: Choose appropriate containers

## 🏆 Final Tips

1. **Practice Daily**: Code at least one problem every day
2. **Read Code**: Study open-source C++ projects
3. **Understand Internals**: Know how STL containers work internally
4. **Stay Updated**: Follow C++ standards evolution
5. **Mock Interviews**: Practice explaining your thought process

---

**Remember: The goal is not just to solve the problem, but to demonstrate clean, efficient, and maintainable C++ code! 🚀**
