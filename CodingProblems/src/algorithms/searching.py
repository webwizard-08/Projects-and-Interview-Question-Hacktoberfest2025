"""
This file contains solutions to searching-related coding problems.
"""

def two_sum(nums, target):
    """
    Given an array of integers `nums` and an integer `target`, return the indices of the two numbers such that they add up to `target`.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    Args:
        nums (list[int]): A list of integers.
        target (int): The target sum.

    Returns:
        list[int]: The indices of the two numbers that add up to the target, or None if no solution is found.

    Example:
        >>> two_sum([2, 7, 11, 15], 9)
        [0, 1]
        >>> two_sum([3, 2, 4], 6)
        [1, 2]
    """
    num_to_index = {}  # Dictionary to store number and its index

    for index, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], index]
        num_to_index[num] = index

    return None  # Return None if no solution is found

# Explanation:
# The `two_sum` function finds two numbers in a list that add up to a given target.
# It uses a dictionary (hash map) to achieve a time complexity of O(n), where n is the number of elements in the list.
#
# How it works:
# 1. We create a dictionary `num_to_index` to store each number and its index as we iterate through the list.
# 2. For each number `num` in the list, we calculate its `complement` (the other number that would add up to the target).
# 3. We check if the `complement` is already in our dictionary.
#    - If it is, we have found our pair. We return the index of the `complement` (from the dictionary) and the index of the current `num`.
#    - If it is not, we add the current `num` and its index to the dictionary.
# 4. If we finish iterating through the list and haven't found a solution, we return `None`.
#
# Time Complexity: O(n) because we iterate through the list of numbers once.
# Space Complexity: O(n) because, in the worst case, we might store all the numbers in the dictionary.
