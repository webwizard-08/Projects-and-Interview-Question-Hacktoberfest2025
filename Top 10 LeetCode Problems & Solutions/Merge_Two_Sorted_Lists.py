# ==================================================
#                LeetCode: 21
#            Merge Two Sorted Lists
# ==================================================
"""
Problem Statement:
You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.
"""

# Link: https://leetcode.com/problems/merge-two-sorted-lists/description/

""" 
Constraints:
The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both list1 and list2 are sorted in non-decreasing order.                      
"""

#Explanation:
# Use a dummy node to build new list.
# Attach the smaller node each time.
# Append remaining nodes after one list ends.

#Solution:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def mergeTwoLists(l1, l2):
    dummy = tail = ListNode()
    while l1 and l2:
        if l1.val < l2.val:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next
"""
Complexiety Analysis:
Time: O(n + m)
Space: O(1)
"""

"""
Example 1:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Example 2:
Input: list1 = [], list2 = []
Output: []

Example 3:
Input: list1 = [], list2 = [0]
Output: [0]
"""