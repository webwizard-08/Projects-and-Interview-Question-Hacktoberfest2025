"""
Common Interview Coding Patterns and Techniques
Author: AI Assistant
Description:
This file contains common coding patterns, techniques, and solutions
frequently asked in technical interviews. Each pattern includes
explanations, examples, and multiple problem variations.
"""

from collections import defaultdict, deque
from typing import List, Optional

# --------------------
# TWO POINTERS PATTERN
# --------------------

def two_sum_sorted(arr: List[int], target: int) -> List[int]:
    """
    Two Sum in Sorted Array using Two Pointers
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]

def remove_duplicates(arr: List[int]) -> int:
    """
    Remove Duplicates from Sorted Array
    Time: O(n), Space: O(1)
    """
    if not arr:
        return 0
    
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    return slow + 1

def container_with_most_water(heights: List[int]) -> int:
    """
    Container With Most Water
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(heights) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        height = min(heights[left], heights[right])
        max_area = max(max_area, width * height)
        
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

# --------------------
# SLIDING WINDOW PATTERN
# --------------------

def max_sum_subarray(arr: List[int], k: int) -> int:
    """
    Maximum Sum of Subarray of Size K
    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return -1
    
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

def longest_substring_without_repeating(s: str) -> int:
    """
    Longest Substring Without Repeating Characters
    Time: O(n), Space: O(min(m,n))
    """
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

def find_anagrams(s: str, p: str) -> List[int]:
    """
    Find All Anagrams in a String
    Time: O(n), Space: O(1)
    """
    if len(p) > len(s):
        return []
    
    p_count = defaultdict(int)
    s_count = defaultdict(int)
    
    for char in p:
        p_count[char] += 1
    
    for i in range(len(p)):
        s_count[s[i]] += 1
    
    result = []
    if p_count == s_count:
        result.append(0)
    
    for i in range(len(p), len(s)):
        s_count[s[i]] += 1
        s_count[s[i - len(p)]] -= 1
        
        if s_count[s[i - len(p)]] == 0:
            del s_count[s[i - len(p)]]
        
        if p_count == s_count:
            result.append(i - len(p) + 1)
    
    return result

# --------------------
# FAST AND SLOW POINTERS (FLOYD'S CYCLE)
# --------------------

def has_cycle(head) -> bool:
    """
    Linked List Cycle Detection
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next
    
    while fast and fast.next:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next
    
    return False

def find_cycle_start(head):
    """
    Find the Start of Cycle in Linked List
    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None
    
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

def find_middle_node(head):
    """
    Find Middle Node of Linked List
    Time: O(n), Space: O(1)
    """
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# --------------------
# MERGE INTERVALS PATTERN
# --------------------

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge Overlapping Intervals
    Time: O(n log n), Space: O(1)
    """
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged

def insert_interval(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    """
    Insert New Interval
    Time: O(n), Space: O(1)
    """
    result = []
    i = 0
    
    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    while i < len(intervals) and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    result.append(new_interval)
    
    while i < len(intervals):
        result.append(intervals[i])
        i += 1
    
    return result

# --------------------
# CYCLIC SORT PATTERN
# --------------------

def find_missing_number(nums: List[int]) -> int:
    """
    Find Missing Number in Array (0 to n)
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum

def find_all_missing_numbers(nums: List[int]) -> List[int]:
    """
    Find All Missing Numbers in Array (1 to n)
    Time: O(n), Space: O(1)
    """
    i = 0
    while i < len(nums):
        j = nums[i] - 1
        if nums[i] != nums[j]:
            nums[i], nums[j] = nums[j], nums[i]
        else:
            i += 1
    
    missing = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            missing.append(i + 1)
    
    return missing

def find_duplicate_number(nums: List[int]) -> int:
    """
    Find Duplicate Number in Array
    Time: O(n), Space: O(1)
    """
    slow = fast = nums[0]
    
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow

# --------------------
# IN-PLACE REVERSAL OF LINKED LIST
# --------------------

def reverse_linked_list(head):
    """
    Reverse Linked List
    Time: O(n), Space: O(1)
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

def reverse_linked_list_between(head, left: int, right: int):
    """
    Reverse Linked List Between Positions
    Time: O(n), Space: O(1)
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    for _ in range(left - 1):
        prev = prev.next
    
    current = prev.next
    
    for _ in range(right - left):
        next_temp = current.next
        current.next = next_temp.next
        next_temp.next = prev.next
        prev.next = next_temp
    
    return dummy.next

# --------------------
# TREE BFS PATTERN
# --------------------

def level_order_traversal(root) -> List[List[int]]:
    """
    Binary Tree Level Order Traversal
    Time: O(n), Space: O(w) where w is max width
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result

def zigzag_level_order(root) -> List[List[int]]:
    """
    Binary Tree Zigzag Level Order Traversal
    Time: O(n), Space: O(w)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if left_to_right:
                level.append(node.val)
            else:
                level.insert(0, node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
        left_to_right = not left_to_right
    
    return result

# --------------------
# TREE DFS PATTERN
# --------------------

def max_depth_binary_tree(root) -> int:
    """
    Maximum Depth of Binary Tree
    Time: O(n), Space: O(h) where h is height
    """
    if not root:
        return 0
    
    left_depth = max_depth_binary_tree(root.left)
    right_depth = max_depth_binary_tree(root.right)
    
    return max(left_depth, right_depth) + 1

def path_sum(root, target_sum: int) -> bool:
    """
    Path Sum in Binary Tree
    Time: O(n), Space: O(h)
    """
    if not root:
        return False
    
    if not root.left and not root.right:
        return root.val == target_sum
    
    remaining_sum = target_sum - root.val
    return (path_sum(root.left, remaining_sum) or 
            path_sum(root.right, remaining_sum))

def find_paths_with_sum(root, target_sum: int) -> List[List[int]]:
    """
    Find All Paths with Target Sum
    Time: O(n^2), Space: O(h)
    """
    def dfs(node, remaining_sum, current_path, all_paths):
        if not node:
            return
        
        current_path.append(node.val)
        
        if not node.left and not node.right and remaining_sum == node.val:
            all_paths.append(current_path[:])
        else:
            dfs(node.left, remaining_sum - node.val, current_path, all_paths)
            dfs(node.right, remaining_sum - node.val, current_path, all_paths)
        
        current_path.pop()
    
    all_paths = []
    dfs(root, target_sum, [], all_paths)
    return all_paths

# --------------------
# SUBSETS PATTERN
# --------------------

def generate_subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate All Subsets
    Time: O(2^n), Space: O(2^n)
    """
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    result = []
    backtrack(0, [])
    return result

def generate_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate All Permutations
    Time: O(n!), Space: O(n!)
    """
    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation[:])
            return
        
        for num in nums:
            if num not in current_permutation:
                current_permutation.append(num)
                backtrack(current_permutation)
                current_permutation.pop()
    
    result = []
    backtrack([])
    return result

# --------------------
# MODIFIED BINARY SEARCH
# --------------------

def search_rotated_array(nums: List[int], target: int) -> int:
    """
    Search in Rotated Sorted Array
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

def find_peak_element(nums: List[int]) -> int:
    """
    Find Peak Element
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left

# --------------------
# TOP K ELEMENTS PATTERN
# --------------------

def find_k_largest(nums: List[int], k: int) -> List[int]:
    """
    Find K Largest Elements
    Time: O(n log k), Space: O(k)
    """
    import heapq
    
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Top K Frequent Elements
    Time: O(n log k), Space: O(n)
    """
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

def k_closest_points(points: List[List[int]], k: int) -> List[List[int]]:
    """
    K Closest Points to Origin
    Time: O(n log k), Space: O(k)
    """
    import heapq
    
    def distance(point):
        return point[0]**2 + point[1]**2
    
    heap = []
    for point in points:
        heapq.heappush(heap, (-distance(point), point))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [point for _, point in heap]

# --------------------
# K-WAY MERGE PATTERN
# --------------------

def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge K Sorted Linked Lists
    Time: O(n log k), Space: O(k)
    """
    import heapq
    
    heap = []
    dummy = ListNode(0)
    current = dummy
    
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next

def find_kth_smallest_in_sorted_matrix(matrix: List[List[int]], k: int) -> int:
    """
    Find Kth Smallest Element in Sorted Matrix
    Time: O(k log n), Space: O(n)
    """
    import heapq
    
    heap = []
    n = len(matrix)
    
    for i in range(n):
        heapq.heappush(heap, (matrix[i][0], i, 0))
    
    for _ in range(k - 1):
        val, row, col = heapq.heappop(heap)
        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))
    
    return heap[0][0]

# --------------------
# DYNAMIC PROGRAMMING PATTERNS
# --------------------

def fibonacci_dp(n: int) -> int:
    """
    Fibonacci with Dynamic Programming
    Time: O(n), Space: O(1)
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Longest Increasing Subsequence
    Time: O(n^2), Space: O(n)
    """
    if not nums:
        return 0
    
    dp = [1] * len(nums)
    
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def edit_distance(word1: str, word2: str) -> int:
    """
    Edit Distance (Levenshtein Distance)
    Time: O(m*n), Space: O(m*n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

# --------------------
# BIT MANIPULATION PATTERNS
# --------------------

def single_number(nums: List[int]) -> int:
    """
    Single Number (XOR Pattern)
    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def missing_number_xor(nums: List[int]) -> int:
    """
    Missing Number using XOR
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    xor_all = 0
    xor_nums = 0
    
    for i in range(n + 1):
        xor_all ^= i
    
    for num in nums:
        xor_nums ^= num
    
    return xor_all ^ xor_nums

def count_set_bits(n: int) -> int:
    """
    Count Number of Set Bits (Hamming Weight)
    Time: O(log n), Space: O(1)
    """
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# --------------------
# UTILITY CLASSES
# --------------------

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# --------------------
# EXAMPLE USAGE AND TESTING
# --------------------

def run_examples():
    """Run examples for various patterns"""
    
    print("=== Two Pointers Examples ===")
    arr = [2, 7, 11, 15]
    print(f"Two Sum in {arr} for target 9: {two_sum_sorted(arr, 9)}")
    
    print("\n=== Sliding Window Examples ===")
    arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
    print(f"Max sum of subarray of size 4: {max_sum_subarray(arr, 4)}")
    
    print("\n=== Fast and Slow Pointers Examples ===")
    # Create a linked list: 1 -> 2 -> 3 -> 4 -> 5
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = ListNode(4)
    head.next.next.next.next = ListNode(5)
    
    middle = find_middle_node(head)
    print(f"Middle node value: {middle.val if middle else None}")
    
    print("\n=== Merge Intervals Examples ===")
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print(f"Merged intervals: {merge_intervals(intervals)}")
    
    print("\n=== Cyclic Sort Examples ===")
    nums = [3, 0, 1]
    print(f"Missing number in {nums}: {find_missing_number(nums)}")
    
    print("\n=== Subsets Examples ===")
    nums = [1, 2, 3]
    print(f"All subsets of {nums}: {generate_subsets(nums)}")
    
    print("\n=== Top K Elements Examples ===")
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Top 3 largest in {nums}: {find_k_largest(nums, 3)}")
    
    print("\n=== Dynamic Programming Examples ===")
    print(f"Fibonacci(10): {fibonacci_dp(10)}")
    
    print("\n=== Bit Manipulation Examples ===")
    nums = [2, 2, 1]
    print(f"Single number in {nums}: {single_number(nums)}")

if __name__ == "__main__":
    run_examples()
