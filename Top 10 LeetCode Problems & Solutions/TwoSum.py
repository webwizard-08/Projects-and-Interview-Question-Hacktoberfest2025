# ==================================================
#                LeetCode: 1
#                  Two Sum
# ==================================================
"""
Problem Statement:
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
"""

# Link: https://leetcode.com/problems/two-sum/description/

"""
Constraints:
2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.                      
"""

# Explanation:
# Use a dictionary to store each number’s index.
# For each n, check if target - n exists in seen.
# Return indices when found — avoids nested loops.

#Solution:
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in seen:
            return [seen[diff], i]
        seen[n] = i
"""
Complexiety Analysis:
Time: O(n)
Space: O(n)
"""

"""
Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]
"""
