# ==================================================
#                LeetCode: 49
#               Group Anagrams
# ==================================================
"""
Problem Statement:
Given an array of strings strs, group the anagrams together. You can return the answer in any order.
"""

# Link: https://leetcode.com/problems/group-anagrams/description/

"""
Constraints:
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.                     
"""

#Explanation:
# Sort each word → same anagrams share sorted key.
# Group by key using dictionary.

#Solution:
from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())

"""
Complexiety Analysis:
Time: O(n * k log k) (k = word length)
Space: O(n * k)
"""

"""
Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Explanation:
There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]
"""