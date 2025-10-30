"""
Pattern Matching Algorithms Implementation

This module implements various pattern matching algorithms used in string processing
and bioinformatics. Each algorithm has its own use case and performance characteristics.

Implemented Algorithms:
1. KMP (Knuth-Morris-Pratt)
2. Rabin-Karp
3. Z Algorithm
"""

def compute_lps_array(pattern):
    """
    Compute Longest Proper Prefix which is also Suffix array.
    Used in KMP algorithm.
    
    Time Complexity: O(m) where m is length of pattern
    Space Complexity: O(m)
    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    return lps

def kmp_search(text, pattern):
    """
    Knuth-Morris-Pratt algorithm for pattern matching.
    
    Time Complexity: O(n + m) where n is text length and m is pattern length
    Space Complexity: O(m)
    
    Args:
        text: String to search in
        pattern: Pattern to search for
        
    Returns:
        List of starting indices where pattern is found
    """
    if not pattern or not text:
        return []
    
    matches = []
    n = len(text)
    m = len(pattern)
    
    # Compute LPS array
    lps = compute_lps_array(pattern)
    
    i = 0  # index for text
    j = 0  # index for pattern
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

def rabin_karp_search(text, pattern, prime=101):
    """
    Rabin-Karp algorithm using rolling hash.
    
    Time Complexity: O(n + m) average case, O(nm) worst case
    Space Complexity: O(1)
    
    Args:
        text: String to search in
        pattern: Pattern to search for
        prime: Prime number for hash function
        
    Returns:
        List of starting indices where pattern is found
    """
    if not pattern or not text:
        return []
    
    matches = []
    n = len(text)
    m = len(pattern)
    if m > n:
        return matches
    
    # Hash values
    pattern_hash = 0
    text_hash = 0
    
    # Hash multiplier
    h = pow(256, m - 1) % prime
    
    # Calculate initial hash values
    for i in range(m):
        pattern_hash = (256 * pattern_hash + ord(pattern[i])) % prime
        text_hash = (256 * text_hash + ord(text[i])) % prime
    
    # Slide pattern over text
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            # Check character by character
            if text[i:i + m] == pattern:
                matches.append(i)
        
        # Calculate hash value for next window
        if i < n - m:
            text_hash = (256 * (text_hash - ord(text[i]) * h) + 
                        ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime
    
    return matches

def compute_z_array(string):
    """
    Compute Z array for Z algorithm.
    Z[i] is the length of the longest substring starting from str[i]
    that is also a prefix of str.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    n = len(string)
    z = [0] * n
    left = right = 0
    
    for i in range(1, n):
        if i > right:
            left = right = i
            while right < n and string[right - left] == string[right]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            k = i - left
            if z[k] < right - i + 1:
                z[i] = z[k]
            else:
                left = i
                while right < n and string[right - left] == string[right]:
                    right += 1
                z[i] = right - left
                right -= 1
    
    return z

def z_algorithm(text, pattern):
    """
    Z Algorithm for pattern matching.
    
    Time Complexity: O(n + m)
    Space Complexity: O(n + m)
    
    Args:
        text: String to search in
        pattern: Pattern to search for
        
    Returns:
        List of starting indices where pattern is found
    """
    if not pattern or not text:
        return []
    
    # Concatenate pattern and text with a special character
    concat = pattern + "$" + text
    z = compute_z_array(concat)
    matches = []
    
    # Pattern is found when Z value equals pattern length
    for i in range(len(pattern) + 1, len(concat)):
        if z[i] == len(pattern):
            matches.append(i - len(pattern) - 1)
    
    return matches

def run_tests():
    """Test cases for all implemented algorithms"""
    test_cases = [
        {
            "text": "AABAACAADAABAAABAA",
            "pattern": "AABA",
            "expected": [0, 9, 13]
        },
        {
            "text": "GEEKS FOR GEEKS",
            "pattern": "GEEK",
            "expected": [0, 8]
        },
        {
            "text": "AAAAA",
            "pattern": "AA",
            "expected": [0, 1, 2, 3]
        }
    ]
    
    algorithms = [
        ("KMP", kmp_search),
        ("Rabin-Karp", rabin_karp_search),
        ("Z Algorithm", z_algorithm)
    ]
    
    for test_case in test_cases:
        print(f"\nTest Case: text='{test_case['text']}', "
              f"pattern='{test_case['pattern']}'")
        print(f"Expected: {test_case['expected']}")
        
        for name, algorithm in algorithms:
            result = algorithm(test_case['text'], test_case['pattern'])
            print(f"{name}: {result}")
            assert result == test_case['expected'], \
                f"{name} failed! Expected {test_case['expected']}, got {result}"

if __name__ == "__main__":
    run_tests()
