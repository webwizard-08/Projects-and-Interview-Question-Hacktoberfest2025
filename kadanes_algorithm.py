# Kadane's Algorithm Implementation in Python

def kadanes_algorithm(arr):
    """
    Finds the maximum sum of a contiguous subarray in a given array of numbers.

    The algorithm iterates through the array, keeping track of the maximum sum
    ending at the current position and the overall maximum sum found so far.

    Args:
        arr: A list of integers (can contain positive and negative numbers).

    Returns:
        The integer value of the maximum subarray sum. Returns 0 if the array is empty.
    """
    if not arr:
        return 0

    # Initialize variables
    # max_so_far will store the maximum sum found anywhere in the array
    # current_max will store the maximum sum of the subarray ending at the current position
    max_so_far = arr[0]
    current_max = arr[0]

    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        num = arr[i]
        # Decide whether to extend the existing subarray or start a new one
        # by comparing the current number with the sum of the current number and the previous max subarray
        current_max = max(num, current_max + num)
        
        # Update the overall maximum sum if the current subarray's sum is greater
        max_so_far = max(max_so_far, current_max)

    return max_so_far

# This block will be executed when the script is run directly
if __name__ == "__main__":
    # Dummy data to test the algorithm
    dummy_data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    
    print(f"Original Array: {dummy_data}")
    
    # Execute the algorithm with the dummy data
    max_sum = kadanes_algorithm(dummy_data)
    
    print(f"The maximum subarray sum is: {max_sum}")
    
    # The subarray with the largest sum is [4, -1, 2, 1], which sums to 6.
