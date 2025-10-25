from typing import List

def find_majority_element(nums: List[int]) -> int:
    """
    Finds the majority element in an array.

    The majority element is the element that appears more than ⌊n / 2⌋ times.

    Args:
        nums: A list of integers.

    Returns:
        The majority element.
    """
    candidate = None
    count = 0
    for num in nums:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1
    return candidate
