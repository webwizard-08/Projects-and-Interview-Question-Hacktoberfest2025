def two_sum(nums, target):
    """
    Find indices of the two numbers in the list that add up to the target.
    
    Constraint:
    - nums must be a list of integers
    - target must be an integer
    
    Time Complexity: O(n^2)  # simple approach
    Space Complexity: O(1)
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return None


# Example
print(two_sum([2, 7, 11, 15], 9))  # Output: [0, 1]
