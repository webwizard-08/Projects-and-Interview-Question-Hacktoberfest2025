def counting_sort(arr):
    """
    Counting sort implementation.
    
    Args:
        arr: List of non-negative integers
    
    Returns:
        list: Sorted list of integers
    
    Time Complexity: O(n + k) where n is the number of elements in the input array and k is the range of the input.
    Space Complexity: O(k)
    """
    max_val = max(arr) if arr else 0
    count = [0] * (max_val + 1)
    output = [0] * len(arr)

    for i in arr:
        count[i] += 1

    for i in range(1, len(count)):
        count[i] += count[i-1]

    for i in range(len(arr)-1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1

    return output

# Example usage
if __name__ == "__main__":
    arr = [4, 2, 2, 8, 3, 3, 1]
    sorted_arr = counting_sort(arr)
    print(f"Original array: {arr}")
    print(f"Sorted array: {sorted_arr}")
