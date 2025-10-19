#Top 20 Array Searching Interview Questions in Python
# Description: This file contains 20 essential array searching problems frequently asked in interviews.
# Each problem includes a short description and a Python implementation.

# 1️⃣ Linear Search in an Array
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 2️⃣ Binary Search in a Sorted Array
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 3️⃣ Search in a Rotated Sorted Array
def search_in_rotated_array(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

# 4️⃣ Find First and Last Occurrence of an Element
def find_first_last(arr, target):
    first, last = -1, -1
    for i in range(len(arr)):
        if arr[i] == target:
            if first == -1:
                first = i
            last = i
    return (first, last)

# 5️⃣ Search in a Nearly Sorted Array
def search_nearly_sorted(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        if mid - 1 >= low and arr[mid - 1] == target:
            return mid - 1
        if mid + 1 <= high and arr[mid + 1] == target:
            return mid + 1
        if target < arr[mid]:
            high = mid - 2
        else:
            low = mid + 2
    return -1

# 6️⃣ Find Peak Element in an Array
def find_peak_element(arr):
    low, high = 0, len(arr) - 1
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < arr[mid + 1]:
            low = mid + 1
        else:
            high = mid
    return low

# 7️⃣ Find Minimum Element in Rotated Sorted Array
def find_min_rotated(arr):
    low, high = 0, len(arr) - 1
    while low < high:
        mid = (low + high) // 2
        if arr[mid] > arr[high]:
            low = mid + 1
        else:
            high = mid
    return arr[low]

# 8️⃣ Search in a 2D Matrix
def search_matrix(matrix, target):
    rows, cols = len(matrix), len(matrix[0])
    i, j = 0, cols - 1
    while i < rows and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    return False

# 9️⃣ Count Occurrences of an Element
def count_occurrences(arr, target):
    return arr.count(target)

# 🔟 Search in an Infinite Sorted Array
def search_in_infinite_array(arr, target):
    low, high = 0, 1
    while high < len(arr) and arr[high] < target:
        low = high
        high *= 2
    high = min(high, len(arr) - 1)
    return binary_search(arr[low:high + 1], target)

# 1️⃣1️⃣ Search for Element in a Bitonic Array
def search_bitonic(arr, target):
    peak = find_peak_element(arr)
    left_part = arr[:peak + 1]
    right_part = arr[peak + 1:]
    res = binary_search(left_part, target)
    if res != -1:
        return res
    res = binary_search(right_part[::-1], target)
    return len(arr) - res - 1 if res != -1 else -1

# 1️⃣2️⃣ Find Floor and Ceiling of a Number
def find_floor_ceil(arr, x):
    floor_val, ceil_val = -1, -1
    for num in arr:
        if num <= x:
            floor_val = num
        if num >= x and ceil_val == -1:
            ceil_val = num
    return floor_val, ceil_val

# 1️⃣3️⃣ Search for a Pair with Given Sum
def has_pair_with_sum(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return True
        elif s < target:
            left += 1
        else:
            right -= 1
    return False

# 1️⃣4️⃣ Search for Triplets with Given Sum
def has_triplet_with_sum(arr, target):
    arr.sort()
    for i in range(len(arr) - 2):
        left, right = i + 1, len(arr) - 1
        while left < right:
            s = arr[i] + arr[left] + arr[right]
            if s == target:
                return True
            elif s < target:
                left += 1
            else:
                right -= 1
    return False

# 1️⃣5️⃣ Search using Hashing in Unsorted Array
def search_hashing(arr, target):
    return target in set(arr)

# 1️⃣6️⃣ Search in an Array with Duplicates
def search_with_duplicates(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# 1️⃣7️⃣ Find Majority Element (Moore’s Voting Algorithm)
def majority_element(arr):
    count, candidate = 0, None
    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate

# 1️⃣8️⃣ Find Square Root of a Number using Binary Search
def sqrt_binary_search(n):
    low, high = 0, n
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        if mid * mid == n:
            return mid
        elif mid * mid < n:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    return ans

# 1️⃣9️⃣ Find Local Minima in an Array
def local_minima(arr):
    n = len(arr)
    for i in range(n):
        if (i == 0 or arr[i - 1] > arr[i]) and (i == n - 1 or arr[i] < arr[i + 1]):
            return arr[i]
    return None

# 2️⃣0️⃣ Search for Subarray with Given Sum
def subarray_with_sum(arr, target):
    curr_sum, start = 0, 0
    for end in range(len(arr)):
        curr_sum += arr[end]
        while curr_sum > target:
            curr_sum -= arr[start]
            start += 1
        if curr_sum == target:
            return (start, end)
    return (-1, -1)