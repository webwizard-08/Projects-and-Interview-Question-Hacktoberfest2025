# ==================================================
#                LeetCode: 704
#                Binary Search
# ==================================================
"""
Problem Statement:
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.
You must write an algorithm with O(log n) runtime complexity.
"""

# Link: https://leetcode.com/problems/binary-search/description/

"""
Constraints:
3 <= nums.length <= 3000
-105 <= nums[i] <= 105                         
"""

#Explanation:
# Repeatedly divide the range in half.
# Compare target with middle element.

#Solution:
def search(nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1
"""
Complexiety Analysis:
Time: O(log n)
Space: O(1)
"""

"""
Example 1:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Example 2:
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
"""
