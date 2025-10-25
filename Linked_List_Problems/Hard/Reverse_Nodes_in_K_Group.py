"""
Reverse Nodes in k-Group Problem
Time Complexity: O(n)
Space Complexity: O(1)

Problem Statement:
Given the head of a linked list, reverse the nodes of the list k at a time,
and return the modified list.

- k is a positive integer ≤ length of the linked list.
- If the number of nodes is not a multiple of k, then the last remaining nodes
  should stay in the same order.
- You may not alter the values in the nodes, only the links between nodes.

Example:
Input: head = [1, 2, 3, 4, 5], k = 2
Output: [2, 1, 4, 3, 5]

Input: head = [1, 2, 3, 4, 5], k = 3
Output: [3, 2, 1, 4, 5]
"""

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseKGroup(head, k):
    # Helper to get the kth node from current
    def getKth(curr, k):
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr

    dummy = ListNode(0, head)
    group_prev = dummy

    while True:
        kth = getKth(group_prev, k)
        if not kth:
            break  # fewer than k nodes remain
        group_next = kth.next

        # Reverse the group
        prev, curr = kth.next, group_prev.next
        while curr != group_next:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp

        # Connect reversed group to the rest of the list
        tmp = group_prev.next
        group_prev.next = kth
        group_prev = tmp

    return dummy.next


# Helper functions for testing
def build_linked_list(values):
    head = ListNode(values[0])
    curr = head
    for val in values[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

def print_linked_list(head):
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res


# Test cases
if __name__ == "__main__":
    tests = [
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 3, 4, 5], 3),
        ([1, 2, 3, 4, 5, 6], 2),
        ([1, 2, 3, 4, 5], 1),
        ([1], 1)
    ]
    
    for i, (arr, k) in enumerate(tests, 1):
        head = build_linked_list(arr)
        print(f"Test Case {i}:")
        print(f"Input List: {arr}, k = {k}")
        new_head = reverseKGroup(head, k)
        print(f"Reversed List: {print_linked_list(new_head)}\n")
