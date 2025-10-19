# Top 20 String Searching & Pattern-Matching Interview Questions in Python
# Description: Python implementation of 20 essential string searching and pattern-matching problems,

# 1️⃣ Naive Substring Search
def naive_substring_search(text, pattern):
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            return i
    return -1

# 2️⃣ Knuth-Morris-Pratt (KMP) Algorithm
def kmp_search(text, pattern):
    def compute_lps(pat):
        lps = [0]*len(pat)
        length = 0
        i = 1
        while i < len(pat):
            if pat[i] == pat[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length-1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(pattern)
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                return i - j
        else:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return -1

# 3️⃣ Rabin-Karp Algorithm (Rolling Hash)
def rabin_karp_search(text, pattern):
    d = 256
    q = 101  # prime number
    n, m = len(text), len(pattern)
    h = pow(d, m-1) % q
    p = t = 0
    for i in range(m):
        p = (d*p + ord(pattern[i])) % q
        t = (d*t + ord(text[i])) % q
    for i in range(n - m + 1):
        if p == t:
            if text[i:i+m] == pattern:
                return i
        if i < n - m:
            t = (d*(t - ord(text[i])*h) + ord(text[i+m])) % q
            t = (t + q) % q
    return -1

# 4️⃣ Boyer-Moore String Search
def boyer_moore_search(text, pattern):
    m = len(pattern)
    n = len(text)
    if m == 0: return 0
    bad_char = [-1]*256
    for i in range(m):
        bad_char[ord(pattern[i])] = i
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s+j]:
            j -= 1
        if j < 0:
            return s
        else:
            s += max(1, j - bad_char[ord(text[s+j])])
    return -1

# 5️⃣ Z-Algorithm for Pattern Matching
def z_algorithm_search(text, pattern):
    concat = pattern + "$" + text
    n = len(concat)
    Z = [0]*n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            Z[i] = min(r-i+1, Z[i-l])
        while i+Z[i] < n and concat[Z[i]] == concat[i+Z[i]]:
            Z[i] += 1
        if i+Z[i]-1 > r:
            l, r = i, i+Z[i]-1
    for i in range(len(pattern)+1, n):
        if Z[i] == len(pattern):
            return i - len(pattern) - 1
    return -1

# 6️⃣ Naive Pattern Search (similar to 1)
def naive_pattern_search(text, pattern):
    return naive_substring_search(text, pattern)

# 7️⃣ Regular Expression Match
import re
def regex_search(text, pattern):
    match = re.search(pattern, text)
    return match.start() if match else -1

# 8️⃣ Search Word in Sentence
def search_word_in_sentence(sentence, word):
    words = sentence.split()
    for i, w in enumerate(words):
        if w == word:
            return i
    return -1

# 9️⃣ Case-Insensitive Search
def case_insensitive_search(text, pattern):
    return text.lower().find(pattern.lower())

# 🔟 Count Occurrences of Substring
def count_substring_occurrences(text, pattern):
    count = start = 0
    while True:
        start = text.find(pattern, start)
        if start == -1:
            break
        count += 1
        start += 1
    return count

# 1️⃣1️⃣ Find Indexes of All Matches
def all_match_indexes(text, pattern):
    indexes = []
    start = 0
    while True:
        start = text.find(pattern, start)
        if start == -1:
            break
        indexes.append(start)
        start += 1
    return indexes

# 1️⃣2️⃣ Search in Character Array
def search_in_char_array(arr, char):
    for i, c in enumerate(arr):
        if c == char:
            return i
    return -1

# 1️⃣3️⃣ Word Boundary Matching
def word_boundary_match(sentence, word):
    words = sentence.split()
    return word in words

# 1️⃣4️⃣ Prefix and Suffix Matching
def prefix_match(text, prefix):
    return text.startswith(prefix)

def suffix_match(text, suffix):
    return text.endswith(suffix)

# 1️⃣5️⃣ Search using indexOf() / lastIndexOf()
def index_of(text, pattern):
    return text.find(pattern)

def last_index_of(text, pattern):
    return text.rfind(pattern)

# 1️⃣6️⃣ Pattern Validation (Regex-Based)
def pattern_validation(text, pattern):
    return bool(re.fullmatch(pattern, text))

# 1️⃣7️⃣ 2D Matrix Word Search (Grid Search)
def grid_search(matrix, word):
    rows, cols = len(matrix), len(matrix[0])
    def search_from(i,j,word_index):
        if word_index == len(word):
            return True
        if i<0 or j<0 or i>=rows or j>=cols or matrix[i][j] != word[word_index]:
            return False
        temp = matrix[i][j]
        matrix[i][j] = '#'
        found = search_from(i+1,j,word_index+1) or search_from(i-1,j,word_index+1) or \
                search_from(i,j+1,word_index+1) or search_from(i,j-1,word_index+1)
        matrix[i][j] = temp
        return found
    for i in range(rows):
        for j in range(cols):
            if search_from(i,j,0):
                return True
    return False
# 1️⃣8️⃣ Palindromic Substring Search
def longest_palindromic_substring(s):
    n = len(s)
    start = max_len = 0
    for i in range(n):
        for j in range(i, n):
            substr = s[i:j+1]
            if substr == substr[::-1] and len(substr) > max_len:
                max_len = len(substr)
                start = i
    return s[start:start+max_len]

# 1️⃣9️⃣ Find All Anagram Occurrences (Sliding Window)
from collections import Counter
def find_anagrams(s, p):
    ns, np = len(s), len(p)
    if ns < np: return []
    p_count = Counter(p)
    s_count = Counter(s[:np])
    result = []
    for i in range(ns-np+1):
        if i != 0:
            s_count[s[i-1]] -= 1
            if s_count[s[i-1]] == 0:
                del s_count[s[i-1]]
            s_count[s[i+np-1]] += 1
        if s_count == p_count:
            result.append(i)
    return result

# 2️⃣0️⃣ Wildcard Search (* and ? matching)
def wildcard_match(s, pattern):
    from fnmatch import fnmatch
    return fnmatch(s, pattern)