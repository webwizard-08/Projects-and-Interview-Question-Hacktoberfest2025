# problem_solving_snippets.py

# Description: Logical coding snippets often seen in coding interviews.

# 1. Check if two strings are anagrams
def is_anagram(s1, s2):
    return sorted(s1) == sorted(s2)

# 2. Find missing number in a sequence
def find_missing_number(nums):
    n = len(nums) + 1
    expected_sum = n * (n + 1) // 2
    return expected_sum - sum(nums)

# 3. Find frequency of each element
def count_frequency(arr):
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq

# 4. Find first non-repeating character
def first_non_repeating_char(s):
    for ch in s:
        if s.count(ch) == 1:
            return ch
    return None

if __name__ == "__main__":
    print("Anagram check:", is_anagram("listen", "silent"))
    print("Missing number:", find_missing_number([1, 2, 3, 5]))
    print("Frequency count:", count_frequency([2, 3, 2, 4, 3, 5]))
    print("First non-repeating char:", first_non_repeating_char("swiss"))
