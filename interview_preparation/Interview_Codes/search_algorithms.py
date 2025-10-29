# search_algorithms.py

# Description: Simple search algorithms often used in interview questions.

# 1. Linear Search
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 2. Binary Search
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

if __name__ == "__main__":
    arr = [2, 4, 6, 8, 10, 12, 14]
    print("Array:", arr)
    print("Linear Search (10):", linear_search(arr, 10))
    print("Binary Search (10):", binary_search(arr, 10))
