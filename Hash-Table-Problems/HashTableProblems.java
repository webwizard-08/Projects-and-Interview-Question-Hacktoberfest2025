/*
 * 🗂️ Hash Table/HashMap Problems - Java Implementation
 * Author: AI Assistant
 * 
 * Description:
 * This Java program contains essential hash table problems frequently asked
 * in technical interviews at top companies like FAANG, Google, Microsoft, and Amazon.
 * 
 * Each problem includes:
 *  - Problem definition
 *  - Clean Java implementation
 *  - Example usage
 *  - Time and Space Complexity
 */

import java.util.*;

public class HashTableProblems {
    
    // 1️⃣ Two Sum - Find two numbers that add up to target
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        return new int[]{};
    }
    
    // 2️⃣ Valid Anagram - Check if two strings are anagrams
    public static boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        
        for (char c : t.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) - 1);
            if (map.get(c) < 0) return false;
        }
        return true;
    }
    
    // 3️⃣ Group Anagrams - Group strings that are anagrams
    public static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            
            map.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
        }
        
        return new ArrayList<>(map.values());
    }
    
    // 4️⃣ First Non-Repeating Character
    public static int firstUniqChar(String s) {
        Map<Character, Integer> map = new HashMap<>();
        
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        
        for (int i = 0; i < s.length(); i++) {
            if (map.get(s.charAt(i)) == 1) {
                return i;
            }
        }
        return -1;
    }
    
    // 5️⃣ Majority Element - Boyer-Moore Voting Algorithm
    public static int majorityElement(int[] nums) {
        int candidate = nums[0];
        int count = 1;
        
        for (int i = 1; i < nums.length; i++) {
            if (count == 0) {
                candidate = nums[i];
                count = 1;
            } else if (candidate == nums[i]) {
                count++;
            } else {
                count--;
            }
        }
        return candidate;
    }
    
    // 6️⃣ Top K Frequent Elements
    public static int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>(
            (a, b) -> a.getValue() - b.getValue()
        );
        
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            pq.offer(entry);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        
        int[] result = new int[k];
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pq.poll().getKey();
        }
        return result;
    }
    
    // 7️⃣ Sort Characters by Frequency
    public static String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>(
            (a, b) -> b.getValue() - a.getValue()
        );
        
        pq.addAll(map.entrySet());
        
        StringBuilder result = new StringBuilder();
        while (!pq.isEmpty()) {
            Map.Entry<Character, Integer> entry = pq.poll();
            for (int i = 0; i < entry.getValue(); i++) {
                result.append(entry.getKey());
            }
        }
        return result.toString();
    }
    
    // 8️⃣ Find All Anagrams in String
    public static List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        if (s.length() < p.length()) return result;
        
        Map<Character, Integer> pMap = new HashMap<>();
        for (char c : p.toCharArray()) {
            pMap.put(c, pMap.getOrDefault(c, 0) + 1);
        }
        
        Map<Character, Integer> sMap = new HashMap<>();
        int windowSize = p.length();
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            sMap.put(c, sMap.getOrDefault(c, 0) + 1);
            
            if (i >= windowSize) {
                char leftChar = s.charAt(i - windowSize);
                sMap.put(leftChar, sMap.get(leftChar) - 1);
                if (sMap.get(leftChar) == 0) {
                    sMap.remove(leftChar);
                }
            }
            
            if (sMap.equals(pMap)) {
                result.add(i - windowSize + 1);
            }
        }
        
        return result;
    }
    
    // 9️⃣ Longest Substring Without Repeating Characters
    public static int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0, maxLen = 0;
        
        for (int right = 0; right < s.length(); right++) {
            while (set.contains(s.charAt(right))) {
                set.remove(s.charAt(left));
                left++;
            }
            set.add(s.charAt(right));
            maxLen = Math.max(maxLen, right - left + 1);
        }
        
        return maxLen;
    }
    
    // 🔟 LRU Cache Implementation
    static class LRUCache {
        class Node {
            int key, value;
            Node prev, next;
            Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }
        
        private int capacity;
        private Map<Integer, Node> map;
        private Node head, tail;
        
        public LRUCache(int capacity) {
            this.capacity = capacity;
            map = new HashMap<>();
            head = new Node(0, 0);
            tail = new Node(0, 0);
            head.next = tail;
            tail.prev = head;
        }
        
        public int get(int key) {
            if (map.containsKey(key)) {
                Node node = map.get(key);
                remove(node);
                addToHead(node);
                return node.value;
            }
            return -1;
        }
        
        public void put(int key, int value) {
            if (map.containsKey(key)) {
                Node node = map.get(key);
                node.value = value;
                remove(node);
                addToHead(node);
            } else {
                if (map.size() >= capacity) {
                    map.remove(tail.prev.key);
                    remove(tail.prev);
                }
                Node newNode = new Node(key, value);
                map.put(key, newNode);
                addToHead(newNode);
            }
        }
        
        private void addToHead(Node node) {
            node.prev = head;
            node.next = head.next;
            head.next.prev = node;
            head.next = node;
        }
        
        private void remove(Node node) {
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }
    }
    
    // 1️⃣1️⃣ Design HashMap
    static class MyHashMap {
        private static final int SIZE = 1000;
        private List<Entry>[] buckets;
        
        class Entry {
            int key, value;
            Entry(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }
        
        public MyHashMap() {
            buckets = new List[SIZE];
            for (int i = 0; i < SIZE; i++) {
                buckets[i] = new ArrayList<>();
            }
        }
        
        public void put(int key, int value) {
            int index = key % SIZE;
            List<Entry> bucket = buckets[index];
            
            for (Entry entry : bucket) {
                if (entry.key == key) {
                    entry.value = value;
                    return;
                }
            }
            bucket.add(new Entry(key, value));
        }
        
        public int get(int key) {
            int index = key % SIZE;
            List<Entry> bucket = buckets[index];
            
            for (Entry entry : bucket) {
                if (entry.key == key) {
                    return entry.value;
                }
            }
            return -1;
        }
        
        public void remove(int key) {
            int index = key % SIZE;
            List<Entry> bucket = buckets[index];
            
            for (int i = 0; i < bucket.size(); i++) {
                if (bucket.get(i).key == key) {
                    bucket.remove(i);
                    return;
                }
            }
        }
    }
    
    // 1️⃣2️⃣ Design HashSet
    static class MyHashSet {
        private static final int SIZE = 1000;
        private List<Integer>[] buckets;
        
        public MyHashSet() {
            buckets = new List[SIZE];
            for (int i = 0; i < SIZE; i++) {
                buckets[i] = new ArrayList<>();
            }
        }
        
        public void add(int key) {
            int index = key % SIZE;
            if (!buckets[index].contains(key)) {
                buckets[index].add(key);
            }
        }
        
        public void remove(int key) {
            int index = key % SIZE;
            buckets[index].removeIf(k -> k == key);
        }
        
        public boolean contains(int key) {
            int index = key % SIZE;
            return buckets[index].contains(key);
        }
    }
    
    // 1️⃣3️⃣ Insert Delete GetRandom O(1)
    static class RandomizedSet {
        private List<Integer> list;
        private Map<Integer, Integer> map;
        private Random random;
        
        public RandomizedSet() {
            list = new ArrayList<>();
            map = new HashMap<>();
            random = new Random();
        }
        
        public boolean insert(int val) {
            if (map.containsKey(val)) return false;
            
            map.put(val, list.size());
            list.add(val);
            return true;
        }
        
        public boolean remove(int val) {
            if (!map.containsKey(val)) return false;
            
            int index = map.get(val);
            int lastElement = list.get(list.size() - 1);
            
            list.set(index, lastElement);
            map.put(lastElement, index);
            
            list.remove(list.size() - 1);
            map.remove(val);
            return true;
        }
        
        public int getRandom() {
            return list.get(random.nextInt(list.size()));
        }
    }
    
    // 1️⃣4️⃣ Word Pattern
    public static boolean wordPattern(String pattern, String s) {
        String[] words = s.split(" ");
        if (pattern.length() != words.length) return false;
        
        Map<Character, String> charToWord = new HashMap<>();
        Map<String, Character> wordToChar = new HashMap<>();
        
        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            String word = words[i];
            
            if (charToWord.containsKey(c) && !charToWord.get(c).equals(word)) {
                return false;
            }
            if (wordToChar.containsKey(word) && wordToChar.get(word) != c) {
                return false;
            }
            
            charToWord.put(c, word);
            wordToChar.put(word, c);
        }
        
        return true;
    }
    
    // 1️⃣5️⃣ Isomorphic Strings
    public static boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) return false;
        
        Map<Character, Character> map = new HashMap<>();
        Set<Character> used = new HashSet<>();
        
        for (int i = 0; i < s.length(); i++) {
            char c1 = s.charAt(i);
            char c2 = t.charAt(i);
            
            if (map.containsKey(c1)) {
                if (map.get(c1) != c2) return false;
            } else {
                if (used.contains(c2)) return false;
                map.put(c1, c2);
                used.add(c2);
            }
        }
        
        return true;
    }
    
    // 1️⃣6️⃣ Minimum Window Substring
    public static String minWindow(String s, String t) {
        Map<Character, Integer> tMap = new HashMap<>();
        for (char c : t.toCharArray()) {
            tMap.put(c, tMap.getOrDefault(c, 0) + 1);
        }
        
        int left = 0, right = 0;
        int minLen = Integer.MAX_VALUE;
        int minStart = 0;
        int count = tMap.size();
        
        while (right < s.length()) {
            char c = s.charAt(right);
            if (tMap.containsKey(c)) {
                tMap.put(c, tMap.get(c) - 1);
                if (tMap.get(c) == 0) count--;
            }
            
            while (count == 0) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    minStart = left;
                }
                
                char leftChar = s.charAt(left);
                if (tMap.containsKey(leftChar)) {
                    tMap.put(leftChar, tMap.get(leftChar) + 1);
                    if (tMap.get(leftChar) > 0) count++;
                }
                left++;
            }
            right++;
        }
        
        return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
    }
    
    // 1️⃣7️⃣ Contains Duplicate
    public static boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) return true;
            set.add(num);
        }
        return false;
    }
    
    // 1️⃣8️⃣ Contains Duplicate II
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i]) && i - map.get(nums[i]) <= k) {
                return true;
            }
            map.put(nums[i], i);
        }
        return false;
    }
    
    // 1️⃣9️⃣ Intersection of Two Arrays
    public static int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();
        
        for (int num : nums1) set1.add(num);
        for (int num : nums2) set2.add(num);
        
        set1.retainAll(set2);
        return set1.stream().mapToInt(i -> i).toArray();
    }
    
    // 2️⃣0️⃣ Intersection of Two Arrays II
    public static int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums1) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        
        List<Integer> result = new ArrayList<>();
        for (int num : nums2) {
            if (map.containsKey(num) && map.get(num) > 0) {
                result.add(num);
                map.put(num, map.get(num) - 1);
            }
        }
        
        return result.stream().mapToInt(i -> i).toArray();
    }
    
    // 🧭 Demonstration
    public static void main(String[] args) {
        System.out.println("=== Hash Table Problems Demo ===");
        
        // Two Sum
        int[] nums = {2, 7, 11, 15};
        int target = 9;
        System.out.println("Two Sum: " + Arrays.toString(twoSum(nums, target)));
        
        // Valid Anagram
        System.out.println("Valid Anagram: " + isAnagram("listen", "silent"));
        
        // Group Anagrams
        String[] strs = {"eat", "tea", "tan", "ate", "nat", "bat"};
        System.out.println("Group Anagrams: " + groupAnagrams(strs));
        
        // First Non-Repeating Character
        System.out.println("First Non-Repeating Character: " + firstUniqChar("leetcode"));
        
        // Majority Element
        int[] majority = {3, 2, 3};
        System.out.println("Majority Element: " + majorityElement(majority));
        
        // Top K Frequent Elements
        int[] topK = {1, 1, 1, 2, 2, 3};
        System.out.println("Top K Frequent: " + Arrays.toString(topKFrequent(topK, 2)));
        
        // Sort Characters by Frequency
        System.out.println("Frequency Sort: " + frequencySort("tree"));
        
        // Longest Substring Without Repeating Characters
        System.out.println("Longest Substring: " + lengthOfLongestSubstring("abcabcbb"));
        
        // Word Pattern
        System.out.println("Word Pattern: " + wordPattern("abba", "dog cat cat dog"));
        
        // Isomorphic Strings
        System.out.println("Isomorphic Strings: " + isIsomorphic("egg", "add"));
        
        // Contains Duplicate
        System.out.println("Contains Duplicate: " + containsDuplicate(new int[]{1, 2, 3, 1}));
        
        // LRU Cache Demo
        LRUCache lru = new LRUCache(2);
        lru.put(1, 1);
        lru.put(2, 2);
        System.out.println("LRU Get 1: " + lru.get(1));
        lru.put(3, 3);
        System.out.println("LRU Get 2: " + lru.get(2));
        
        // RandomizedSet Demo
        RandomizedSet rs = new RandomizedSet();
        rs.insert(1);
        rs.insert(2);
        System.out.println("RandomizedSet GetRandom: " + rs.getRandom());
    }
}
