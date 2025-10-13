# Sort Colors (Dutch National Flag)

# Problem Statement:
# Given an array nums with values 0, 1, and 2, sort them in-place so that all 0’s come first, then 1’s, then 2’s.

# Example:

# Input: nums = [2,0,2,1,1,0]  
# Output: [0,0,1,1,2,2]


# Python Code:

def sortColors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1