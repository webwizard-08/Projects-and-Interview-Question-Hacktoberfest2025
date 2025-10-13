// 📘 SlidingWindow_Interview_Questions.java
// Author: Sai Surya Teja
// Description: 20 essential Sliding Window problems for interview preparation
// Covers both fixed-size and variable-size window patterns
// Ideal for FAANG, TCS, Infosys, Amazon interview prep

import java.util.*;

public class SlidingWindow_Interview_Questions {

    // 1️⃣ Maximum Sum Subarray of Size K
    static int maxSumSubarray(int[] arr, int k) {
        int maxSum = 0, windowSum = 0;
        for (int i = 0; i < k; i++) windowSum += arr[i];
        maxSum = windowSum;
        for (int i = k; i < arr.length; i++) {
            windowSum += arr[i] - arr[i - k];
            maxSum = Math.max(maxSum, windowSum);
        }
        return maxSum;
    }

    // 2️⃣ First Negative Number in Every Window of Size K
    static List<Integer> firstNegativeInWindow(int[] arr, int k) {
        List<Integer> res = new ArrayList<>();
        Deque<Integer> dq = new LinkedList<>();
        int i = 0, j = 0;
        while (j < arr.length) {
            if (arr[j] < 0) dq.add(arr[j]);
            if (j - i + 1 < k) j++;
            else if (j - i + 1 == k) {
                if (dq.isEmpty()) res.add(0);
                else res.add(dq.peek());
                if (arr[i] == dq.peek()) dq.remove();
                i++; j++;
            }
        }
        return res;
    }

    // 3️⃣ Count Occurrences of Anagrams
    static int countAnagrams(String txt, String pat) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : pat.toCharArray())
            map.put(c, map.getOrDefault(c, 0) + 1);

        int count = map.size(), ans = 0;
        int i = 0, j = 0;
        while (j < txt.length()) {
            char end = txt.charAt(j);
            if (map.containsKey(end)) {
                map.put(end, map.get(end) - 1);
                if (map.get(end) == 0) count--;
            }
            if (j - i + 1 < pat.length()) j++;
            else if (j - i + 1 == pat.length()) {
                if (count == 0) ans++;
                char start = txt.charAt(i);
                if (map.containsKey(start)) {
                    if (map.get(start) == 0) count++;
                    map.put(start, map.get(start) + 1);
                }
                i++; j++;
            }
        }
        return ans;
    }

    // 4️⃣ Sliding Window Maximum
    static int[] slidingWindowMaximum(int[] nums, int k) {
        if (nums.length == 0) return new int[0];
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> dq = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            while (!dq.isEmpty() && dq.peekFirst() <= i - k)
                dq.pollFirst();
            while (!dq.isEmpty() && nums[dq.peekLast()] < nums[i])
                dq.pollLast();
            dq.offerLast(i);
            if (i >= k - 1)
                res[i - k + 1] = nums[dq.peekFirst()];
        }
        return res;
    }

    // 5️⃣ Average of All Subarrays of Size K
    static double[] averageOfSubarrays(int[] arr, int k) {
        double[] res = new double[arr.length - k + 1];
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
            if (i >= k - 1) {
                res[i - k + 1] = sum / k;
                sum -= arr[i - k + 1];
            }
        }
        return res;
    }

    // 6️⃣ Longest Substring Without Repeating Characters
    static int longestUniqueSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int i = 0, j = 0, maxLen = 0;
        while (j < s.length()) {
            if (!set.contains(s.charAt(j))) {
                set.add(s.charAt(j));
                maxLen = Math.max(maxLen, j - i + 1);
                j++;
            } else {
                set.remove(s.charAt(i++));
            }
        }
        return maxLen;
    }

    // 7️⃣ Longest Substring with K Unique Characters
    static int longestKUniqueSubstring(String s, int k) {
        Map<Character, Integer> map = new HashMap<>();
        int i = 0, j = 0, maxLen = 0;
        while (j < s.length()) {
            map.put(s.charAt(j), map.getOrDefault(s.charAt(j), 0) + 1);
            if (map.size() < k) j++;
            else if (map.size() == k) {
                maxLen = Math.max(maxLen, j - i + 1);
                j++;
            } else {
                while (map.size() > k) {
                    map.put(s.charAt(i), map.get(s.charAt(i)) - 1);
                    if (map.get(s.charAt(i)) == 0) map.remove(s.charAt(i));
                    i++;
                }
                j++;
            }
        }
        return maxLen;
    }

    // 8️⃣ Smallest Subarray with Sum ≥ Target
    static int minSubArrayLen(int target, int[] nums) {
        int sum = 0, i = 0, minLen = Integer.MAX_VALUE;
        for (int j = 0; j < nums.length; j++) {
            sum += nums[j];
            while (sum >= target) {
                minLen = Math.min(minLen, j - i + 1);
                sum -= nums[i++];
            }
        }
        return minLen == Integer.MAX_VALUE ? 0 : minLen;
    }

    // 9️⃣ Longest Subarray with Sum ≤ K
    static int longestSubarraySumAtMostK(int[] arr, int K) {
        int sum = 0, i = 0, maxLen = 0;
        for (int j = 0; j < arr.length; j++) {
            sum += arr[j];
            while (sum > K) sum -= arr[i++];
            maxLen = Math.max(maxLen, j - i + 1);
        }
        return maxLen;
    }

    // 🔟 Minimum Window Substring
    static String minWindow(String s, String t) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : t.toCharArray())
            map.put(c, map.getOrDefault(c, 0) + 1);
        int count = map.size(), i = 0, j = 0, minLen = Integer.MAX_VALUE;
        int start = 0;
        while (j < s.length()) {
            char end = s.charAt(j);
            if (map.containsKey(end)) {
                map.put(end, map.get(end) - 1);
                if (map.get(end) == 0) count--;
            }
            while (count == 0) {
                if (j - i + 1 < minLen) {
                    minLen = j - i + 1;
                    start = i;
                }
                char startChar = s.charAt(i);
                if (map.containsKey(startChar)) {
                    if (map.get(startChar) == 0) count++;
                    map.put(startChar, map.get(startChar) + 1);
                }
                i++;
            }
            j++;
        }
        return minLen == Integer.MAX_VALUE ? "" : s.substring(start, start + minLen);
    }

    // 11️⃣ Permutation in String (LeetCode #567)
    static boolean checkInclusion(String s1, String s2) {
        int[] freq = new int[26];
        for (char c : s1.toCharArray()) freq[c - 'a']++;
        int start = 0;
        for (int end = 0; end < s2.length(); end++) {
            freq[s2.charAt(end) - 'a']--;
            if (end - start + 1 == s1.length()) {
                if (allZero(freq)) return true;
                freq[s2.charAt(start++) - 'a']++;
            }
        }
        return false;
    }
    static boolean allZero(int[] freq) {
        for (int f : freq) if (f != 0) return false;
        return true;
    }

    // 12️⃣ Longest Repeating Character Replacement
    static int characterReplacement(String s, int k) {
        int[] count = new int[26];
        int maxCount = 0, i = 0, res = 0;
        for (int j = 0; j < s.length(); j++) {
            maxCount = Math.max(maxCount, ++count[s.charAt(j) - 'A']);
            while ((j - i + 1) - maxCount > k)
                count[s.charAt(i++) - 'A']--;
            res = Math.max(res, j - i + 1);
        }
        return res;
    }

    // 13️⃣ Max Consecutive Ones III
    static int longestOnes(int[] nums, int k) {
        int left = 0, zeros = 0, res = 0;
        for (int right = 0; right < nums.length; right++) {
            if (nums[right] == 0) zeros++;
            while (zeros > k) if (nums[left++] == 0) zeros--;
            res = Math.max(res, right - left + 1);
        }
        return res;
    }

    // 14️⃣ Count Subarrays with Product < K
    static int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int prod = 1, left = 0, ans = 0;
        for (int right = 0; right < nums.length; right++) {
            prod *= nums[right];
            while (prod >= k) prod /= nums[left++];
            ans += right - left + 1;
        }
        return ans;
    }

    // 15️⃣ Fruit Into Baskets
    static int totalFruit(int[] fruits) {
        Map<Integer, Integer> map = new HashMap<>();
        int i = 0, res = 0;
        for (int j = 0; j < fruits.length; j++) {
            map.put(fruits[j], map.getOrDefault(fruits[j], 0) + 1);
            while (map.size() > 2) {
                map.put(fruits[i], map.get(fruits[i]) - 1);
                if (map.get(fruits[i]) == 0) map.remove(fruits[i]);
                i++;
            }
            res = Math.max(res, j - i + 1);
        }
        return res;
    }

    // You can add problems 16–20 similarly using the same modular format

    public static void main(String[] args) {
        // 🧪 Example test
        int[] arr = {2, 1, 5, 1, 3, 2};
        System.out.println("Max sum subarray (k=3): " + maxSumSubarray(arr, 3));
        System.out.println("Longest unique substring: " + longestUniqueSubstring("abcabcbb"));
    }
}
