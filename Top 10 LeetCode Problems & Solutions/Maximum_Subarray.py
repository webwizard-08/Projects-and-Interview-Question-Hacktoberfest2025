# ==================================================
#                LeetCode: 53
#               Maximum Subarray
# ==================================================
"""
Problem Statement:
Given an integer array nums, find the subarray with the largest sum, and return its sum.
"""

# Link: https://leetcode.com/problems/maximum-subarray/description/

"""
Constraints:
1 <= nums.length <= 105
-104 <= nums[i] <= 104                       
"""

#Explanation:
# Either extend current subarray or start new.
# Track max sum seen so far.

#Solution:
def maxSubArray(nums):
    cur_sum = max_sum = nums[0]
    for n in nums[1:]:
        cur_sum = max(n, cur_sum + n)
        max_sum = max(max_sum, cur_sum)
    return max_sum
"""
Complexiety Analysis:
Time: O(n)
Space: O(1)
"""

"""
Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.

Example 2:
Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
"""
