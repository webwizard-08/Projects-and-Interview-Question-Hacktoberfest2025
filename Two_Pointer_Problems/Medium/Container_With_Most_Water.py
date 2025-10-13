# Container With Most Water

# Problem Statement:
# You are given an integer array height. The width between the lines i and j is (j - i).
# Find the maximum area of water a container can store.

# Example:

# Input: height = [1,8,6,2,5,4,8,3,7]  
# Output: 49


# Python Code:

def maxArea(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        width = right - left
        max_water = max(max_water, min(height[left], height[right]) * width)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water