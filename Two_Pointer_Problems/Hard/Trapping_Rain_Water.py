# Trapping Rain Water

# Problem Statement:
# Given n non-negative integers representing elevation map bars, compute how much water it can trap after raining.

# Example:

# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]  
# Output: 6


# Python Code:

def trap(height):
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water = 0
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water += max(0, left_max - height[left])
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water += max(0, right_max - height[right])
    return water