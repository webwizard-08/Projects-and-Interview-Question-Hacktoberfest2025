# 📚 Counting Sort

Counting sort is an integer sorting algorithm that assumes that each of the *n* input elements is an integer in the range of 0 to *k*, for some integer *k*. When *k* = O(*n*), the sort runs in O(*n*) time.

## 🎯 What is Counting Sort?

Counting sort is a stable, non-comparison-based sorting algorithm. It works by counting the number of occurrences of each distinct element in the input array. This information is then used to place the elements in their correct sorted positions.

## 📁 Problems Covered

### Easy Level
- Sorting an array of integers within a small range

### Medium Level
- Sorting an array of characters
- Counting occurrences of elements

### Hard Level
- Radix Sort (uses counting sort as a subroutine)

## 🛠️ Implementation Languages

- **Python** - Implemented in `counting_sort.py`

## ⏱️ Time Complexities

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Best Case | O(n + k) | When all elements are the same |
| Average Case | O(n + k) | Uniformly distributed elements |
| Worst Case | O(n + k) | When the range of elements is large |

## 🚀 How to Run

### Python
```bash
python3 counting_sort.py
```

## 📖 Learning Resources

- [Counting Sort - GeeksforGeeks](https://www.geeksforgeeks.org/counting-sort/)
- [Counting Sort - Wikipedia](https://en.wikipedia.org/wiki/Counting_sort)

## 🤝 Contributing

Feel free to add more problems, improve existing solutions, or add implementations in other languages!

---

**Happy Coding! 🎉**
