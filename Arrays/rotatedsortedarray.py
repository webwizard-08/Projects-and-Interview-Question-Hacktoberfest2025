def search_rotated_array(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        # Left sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

# Example
arr = [4, 5, 6, 7, 0, 1, 2]
target = 0
index = search_rotated_array(arr, target)
print(f"Element {target} found at index {index}" if index != -1 else "Element not found")
