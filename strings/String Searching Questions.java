import java.util.*;
import java.util.regex.*;

/**
 * String_Searching_Questions.java
 *
 * 20 essential string searching & pattern matching problems with implementations,
 * example usages in main(), and complexity notes.
 *
 * Author: ChatGPT for Surya
 */
public class String_Searching_Questions {

    // 1. Naive substring search (returns first index or -1)
    // Time: O(n*m), Space: O(1)
    public static int naiveSearch(String text, String pattern) {
        int n = text.length(), m = pattern.length();
        for (int i = 0; i <= n - m; i++) {
            int j = 0;
            while (j < m && text.charAt(i + j) == pattern.charAt(j)) j++;
            if (j == m) return i;
        }
        return -1;
    }

    // 2. Knuth-Morris-Pratt (KMP) algorithm
    // Time: O(n + m), Space: O(m)
    public static List<Integer> kmpSearchAll(String text, String pattern) {
        List<Integer> res = new ArrayList<>();
        if (pattern.length() == 0) return res;
        int[] lps = buildLPS(pattern);
        int i = 0, j = 0;
        while (i < text.length()) {
            if (text.charAt(i) == pattern.charAt(j)) {
                i++; j++;
                if (j == pattern.length()) {
                    res.add(i - j);
                    j = lps[j - 1];
                }
            } else if (j > 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
        return res;
    }

    private static int[] buildLPS(String pat) {
        int m = pat.length();
        int[] lps = new int[m];
        int len = 0; // length of previous longest prefix suffix
        int i = 1;
        while (i < m) {
            if (pat.charAt(i) == pat.charAt(len)) {
                lps[i++] = ++len;
            } else if (len > 0) {
                len = lps[len - 1];
            } else {
                lps[i++] = 0;
            }
        }
        return lps;
    }

    // 3. Rabin-Karp (rolling hash) - returns first index or -1
    // Average Time: O(n + m), Worst O(n*m) (hash collisions), Space: O(1)
    public static int rabinKarp(String text, String pattern) {
        int n = text.length(), m = pattern.length();
        if (m > n) return -1;
        long mod = 1000000007L;
        long base = 256L;
        long hashP = 0, hashT = 0, pow = 1;
        for (int i = 0; i < m; i++) {
            hashP = (hashP * base + pattern.charAt(i)) % mod;
            hashT = (hashT * base + text.charAt(i)) % mod;
            if (i > 0) pow = (pow * base) % mod;
        }
        for (int i = 0; i <= n - m; i++) {
            if (hashP == hashT) {
                // verify
                if (text.substring(i, i + m).equals(pattern)) return i;
            }
            if (i < n - m) {
                hashT = (hashT - text.charAt(i) * pow % mod + mod) % mod;
                hashT = (hashT * base + text.charAt(i + m)) % mod;
            }
        }
        return -1;
    }

    // 4. Boyer-Moore (bad character heuristic simplified)
    // Time: O(n + m) average, worst O(n*m)
    public static int boyerMooreSearch(String text, String pattern) {
        int n = text.length(), m = pattern.length();
        if (m == 0) return 0;
        int[] bad = buildBadCharTable(pattern);
        int s = 0;
        while (s <= n - m) {
            int j = m - 1;
            while (j >= 0 && pattern.charAt(j) == text.charAt(s + j)) j--;
            if (j < 0) return s;
            int shift = Math.max(1, j - bad[text.charAt(s + j)]);
            s += shift;
        }
        return -1;
    }

    private static int[] buildBadCharTable(String pat) {
        final int ALPH = 256;
        int[] bad = new int[ALPH];
        Arrays.fill(bad, -1);
        for (int i = 0; i < pat.length(); i++) bad[pat.charAt(i)] = i;
        return bad;
    }

    // 5. Find all occurrences of a substring (using KMP)
    // Time: O(n + m)
    public static List<Integer> findAllOccurrences(String text, String pattern) {
        return kmpSearchAll(text, pattern);
    }

    // 6. Count occurrences of a substring (including overlapping)
    // Time: O(n*m) naive, or O(n + m) with KMP
    public static int countOccurrences(String text, String pattern) {
        return kmpSearchAll(text, pattern).size();
    }

    // 7. Check if word exists in sentence (word boundary match)
    // Time: O(n)
    public static boolean wordExists(String sentence, String word) {
        String regex = "\\b" + Pattern.quote(word) + "\\b";
        Pattern p = Pattern.compile(regex);
        Matcher m = p.matcher(sentence);
        return m.find();
    }

    // 8. Case-insensitive substring search
    // Time: O(n*m) naive, or uses indexOf on lower-case strings
    public static int caseInsensitiveSearch(String text, String pattern) {
        return text.toLowerCase().indexOf(pattern.toLowerCase());
    }

    // 9. Regex search (find first match and groups)
    // Time: depends on regex complexity
    public static List<String> regexFindAll(String text, String regex) {
        List<String> matches = new ArrayList<>();
        Pattern p = Pattern.compile(regex);
        Matcher m = p.matcher(text);
        while (m.find()) {
            matches.add(m.group());
        }
        return matches;
    }

    // 10. First occurrence using indexOf()
    public static int firstIndexOf(String text, String pattern) {
        return text.indexOf(pattern);
    }

    // 11. Last occurrence using lastIndexOf()
    public static int lastIndexOf(String text, String pattern) {
        return text.lastIndexOf(pattern);
    }

    // 12. Find all words starting with a given prefix
    // Time: O(n * averageWordLength)
    public static List<String> wordsStartingWith(String sentence, String prefix) {
        List<String> res = new ArrayList<>();
        String[] words = sentence.split("\\s+");
        for (String w : words) if (w.startsWith(prefix)) res.add(w);
        return res;
    }

    // 13. Longest palindromic substring (expand around center)
    // Time: O(n^2), Space: O(1)
    public static String longestPalindromicSubstring(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start + 1) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private static int expandAroundCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--; right++;
        }
        return right - left - 1;
    }

    // 14. Find all anagram occurrences of a word in a string (sliding window)
    // Time: O(n + k) where k = alphabet size (here 256), Space: O(k)
    public static List<Integer> findAnagrams(String text, String pattern) {
        List<Integer> res = new ArrayList<>();
        int n = text.length(), m = pattern.length();
        if (m > n) return res;
        int[] freq = new int[256];
        for (char c : pattern.toCharArray()) freq[c]++;
        int count = m;
        for (int i = 0; i < n; i++) {
            if (freq[text.charAt(i)] > 0) count--;
            freq[text.charAt(i)]--;
            if (i >= m) {
                freq[text.charAt(i - m)]++;
                if (freq[text.charAt(i - m)] > 0) count++;
            }
            if (count == 0) res.add(i - m + 1);
        }
        return res;
    }

    // 15. Wildcard matching with ? and * using DP
    // ? matches single char, * matches any sequence (including empty)
    // Time: O(n*m), Space: O(n*m)
    public static boolean wildcardMatch(String text, String pattern) {
        int n = text.length(), m = pattern.length();
        boolean[][] dp = new boolean[n + 1][m + 1];
        dp[0][0] = true;
        for (int j = 1; j <= m; j++) {
            if (pattern.charAt(j - 1) == '*') dp[0][j] = dp[0][j - 1];
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                char pc = pattern.charAt(j - 1);
                if (pc == '*') dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                else if (pc == '?' || pc == text.charAt(i - 1)) dp[i][j] = dp[i - 1][j - 1];
                else dp[i][j] = false;
            }
        }
        return dp[n][m];
    }

    // 16. Find longest substring that matches a regex pattern (brute force)
    // Time: O(n^2 * regexMatchCost)
    public static String longestSubstringMatchingRegex(String text, String regex) {
        String best = "";
        Pattern p = Pattern.compile(regex);
        for (int i = 0; i < text.length(); i++) {
            for (int j = i + 1; j <= text.length(); j++) {
                String sub = text.substring(i, j);
                Matcher m = p.matcher(sub);
                if (m.matches() && sub.length() > best.length()) best = sub;
            }
        }
        return best;
    }

    // 17. 2D matrix word search (grid search) - classic LeetCode word search
    // Time: O(n*m*4^L) worst-case where L is word length
    public static boolean existInGrid(char[][] board, String word) {
        int R = board.length, C = board[0].length;
        boolean[][] vis = new boolean[R][C];
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (dfsGrid(board, word, 0, i, j, vis)) return true;
            }
        }
        return false;
    }

    private static boolean dfsGrid(char[][] b, String w, int idx, int r, int c, boolean[][] vis) {
        if (idx == w.length()) return true;
        if (r < 0 || c < 0 || r >= b.length || c >= b[0].length) return false;
        if (vis[r][c] || b[r][c] != w.charAt(idx)) return false;
        vis[r][c] = true;
        int[] dr = {1, -1, 0, 0};
        int[] dc = {0, 0, 1, -1};
        for (int k = 0; k < 4; k++) {
            if (dfsGrid(b, w, idx + 1, r + dr[k], c + dc[k], vis)) return true;
        }
        vis[r][c] = false;
        return false;
    }

    // 18. Validate if a string matches a given regex fully
    // Time: depends on regex
    public static boolean regexValidate(String text, String regex) {
        return Pattern.matches(regex, text);
    }

    // 19. Find overlapping substrings (list of starting indices)
    // Time: O(n*m) naive or O(n + m) with KMP
    public static List<Integer> findOverlapping(String text, String pattern) {
        List<Integer> res = new ArrayList<>();
        int from = 0;
        while (from <= text.length() - pattern.length()) {
            int idx = text.indexOf(pattern, from);
            if (idx == -1) break;
            res.add(idx);
            from = idx + 1; // allow overlap
        }
        return res;
    }

    // 20. Pattern search using sliding window + Rabin-Karp (returns all indices)
    // Time: average O(n + m)
    public static List<Integer> patternSearchRabinKarpAll(String text, String pattern) {
        List<Integer> res = new ArrayList<>();
        int n = text.length(), m = pattern.length();
        if (m > n) return res;
        long mod = 1000000007L;
        long base = 256L;
        long hashP = 0, hashT = 0, pow = 1;
        for (int i = 0; i < m; i++) {
            hashP = (hashP * base + pattern.charAt(i)) % mod;
            hashT = (hashT * base + text.charAt(i)) % mod;
            if (i > 0) pow = (pow * base) % mod;
        }
        for (int i = 0; i <= n - m; i++) {
            if (hashP == hashT) {
                if (text.substring(i, i + m).equals(pattern)) res.add(i);
            }
            if (i < n - m) {
                hashT = (hashT - text.charAt(i) * pow % mod + mod) % mod;
                hashT = (hashT * base + text.charAt(i + m)) % mod;
            }
        }
        return res;
    }

    // ---------------------- Example / Demo in main ----------------------
    public static void main(String[] args) {
        System.out.println("String Searching Questions - Demo Outputs\n");

        String text = "abxabcabcaby";
        String pat = "abcaby";
        System.out.println("1. Naive search: " + naiveSearch(text, pat)); // expect 6

        System.out.println("2. KMP all occurrences of 'abc': " + kmpSearchAll(text, "abc"));

        System.out.println("3. Rabin-Karp first index: " + rabinKarp(text, pat));

        System.out.println("4. Boyer-Moore: " + boyerMooreSearch(text, pat));

        System.out.println("5. Find all occurrences (KMP) of 'ab': " + findAllOccurrences(text, "ab"));

        System.out.println("6. Count occurrences of 'ab': " + countOccurrences(text, "ab"));

        String sentence = "Hello world, welcome to the world of Java.";
        System.out.println("7. Word exists 'world': " + wordExists(sentence, "world"));

        System.out.println("8. Case-insensitive search 'Welcome': " + caseInsensitiveSearch(sentence, "Welcome"));

        System.out.println("9. Regex find all words starting with w: " + regexFindAll(sentence, "\\\b(w\\\\w*)\\\b"));

        System.out.println("10. First indexOf 'world': " + firstIndexOf(sentence, "world"));
        System.out.println("11. Last indexOf 'world': " + lastIndexOf(sentence, "world"));

        System.out.println("12. Words starting with 'wo': " + wordsStartingWith(sentence, "wo"));

        String pal = "babad";
        System.out.println("13. Longest palindromic substring of 'babad': " + longestPalindromicSubstring(pal));

        String text2 = "cbaebabacd";
        String p2 = "abc";
        System.out.println("14. Anagram start indices: " + findAnagrams(text2, p2)); // expect [0,6]

        System.out.println("15. Wildcard match: 'abef' vs 'a*?f' -> " + wildcardMatch("abef", "a*?f"));

        System.out.println("16. Longest substring matching regex (digits+) in 'a1234b56': " + longestSubstringMatchingRegex("a1234b56", "\\\d+"));

        char[][] board = {
            {'A','B','C','E'},
            {'S','F','C','S'},
            {'A','D','E','E'}
        };
        System.out.println("17. Exist 'ABCCED' in grid: " + existInGrid(board, "ABCCED"));

        System.out.println("18. Regex validate email-like: 'test@domain.com' -> " + regexValidate("test@domain.com", "^[A-Za-z0-9+_.-]+@(.+)$"));

        String ovText = "aaaa";
        String ovPat = "aa";
        System.out.println("19. Overlapping occurrences of 'aa' in 'aaaa': " + findOverlapping(ovText, ovPat)); // [0,1,2]

        System.out.println("20. Rabin-Karp all indices of 'abc' in 'ababcabc': " + patternSearchRabinKarpAll("ababcabc", "abc"));

        System.out.println("\nDemo complete. Use individual methods for more tests and complexity analysis comments in code.");
    }
}
