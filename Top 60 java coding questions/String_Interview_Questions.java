/*
 * 🧠 Top 20 String Interview Questions – Java Implementation
 * Author: Sai Surya
 *
 * Description:
 * This Java program contains 20 essential string-based coding problems
 * frequently asked in technical interviews at top companies like FAANG, TCS, Infosys, and Amazon.
 *
 * Each problem includes:
 *  - Problem definition
 *  - Clean Java implementation
 *  - Example I/O
 *  - Time and Space Complexity
 */

import java.util.*;

public class String_Interview_Questions {

    // 1️⃣ Reverse a String
    public static String reverseString(String str) {
        return new StringBuilder(str).reverse().toString(); // O(n)
    }

    // 2️⃣ Check if a String is Palindrome
    public static boolean isPalindrome(String str) {
        int i = 0, j = str.length() - 1;
        while (i < j)
            if (str.charAt(i++) != str.charAt(j--)) return false;
        return true; // O(n)
    }

    // 3️⃣ Count Vowels and Consonants
    public static void countVowelsAndConsonants(String str) {
        int vowels = 0, consonants = 0;
        str = str.toLowerCase();
        for (char c : str.toCharArray()) {
            if (Character.isLetter(c)) {
                if ("aeiou".indexOf(c) != -1)
                    vowels++;
                else
                    consonants++;
            }
        }
        System.out.println("Vowels: " + vowels + ", Consonants: " + consonants); // O(n)
    }

    // 4️⃣ Check if Two Strings are Anagrams
    public static boolean areAnagrams(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        char[] a = s1.toCharArray(), b = s2.toCharArray();
        Arrays.sort(a);
        Arrays.sort(b);
        return Arrays.equals(a, b); // O(n log n)
    }

    // 5️⃣ First Non-Repeating Character
    public static char firstNonRepeatingChar(String str) {
        Map<Character, Integer> map = new LinkedHashMap<>();
        for (char c : str.toCharArray())
            map.put(c, map.getOrDefault(c, 0) + 1);
        for (char c : map.keySet())
            if (map.get(c) == 1)
                return c;
        return '-'; // O(n)
    }

    // 6️⃣ Remove All Duplicates
    public static String removeDuplicates(String str) {
        Set<Character> seen = new LinkedHashSet<>();
        for (char c : str.toCharArray())
            seen.add(c);
        StringBuilder sb = new StringBuilder();
        for (char c : seen) sb.append(c);
        return sb.toString(); // O(n)
    }

    // 7️⃣ Find Most Frequent Character
    public static char mostFrequentChar(String str) {
        Map<Character, Integer> freq = new HashMap<>();
        for (char c : str.toCharArray())
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        char maxChar = ' ';
        int max = 0;
        for (Map.Entry<Character, Integer> e : freq.entrySet())
            if (e.getValue() > max) {
                max = e.getValue();
                maxChar = e.getKey();
            }
        return maxChar; // O(n)
    }

    // 8️⃣ Check if String Contains Only Digits
    public static boolean isNumeric(String str) {
        return str.matches("\\d+"); // O(n)
    }

    // 9️⃣ Convert to Uppercase and Lowercase
    public static void convertCase(String str) {
        System.out.println("Uppercase: " + str.toUpperCase());
        System.out.println("Lowercase: " + str.toLowerCase());
    }

    // 🔟 Count Words in a Sentence
    public static int countWords(String sentence) {
        if (sentence == null || sentence.trim().isEmpty()) return 0;
        return sentence.trim().split("\\s+").length; // O(n)
    }

    // 11️⃣ Find All Substrings
    public static void printAllSubstrings(String str) {
        for (int i = 0; i < str.length(); i++)
            for (int j = i + 1; j <= str.length(); j++)
                System.out.println(str.substring(i, j)); // O(n²)
    }

    // 12️⃣ Check if One String is Rotation of Another
    public static boolean isRotation(String s1, String s2) {
        return s1.length() == s2.length() && (s1 + s1).contains(s2); // O(n)
    }

    // 13️⃣ Remove All Whitespaces
    public static String removeWhitespaces(String str) {
        return str.replaceAll("\\s+", ""); // O(n)
    }

    // 14️⃣ Compress a String (Run-Length Encoding)
    public static String compressString(String str) {
        StringBuilder sb = new StringBuilder();
        int count = 1;
        for (int i = 1; i <= str.length(); i++) {
            if (i == str.length() || str.charAt(i) != str.charAt(i - 1)) {
                sb.append(str.charAt(i - 1)).append(count);
                count = 1;
            } else count++;
        }
        return sb.toString(); // O(n)
    }

    // 15️⃣ Check if Two Strings are Isomorphic
    public static boolean areIsomorphic(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        Map<Character, Character> map = new HashMap<>();
        Set<Character> used = new HashSet<>();
        for (int i = 0; i < s1.length(); i++) {
            char c1 = s1.charAt(i), c2 = s2.charAt(i);
            if (map.containsKey(c1)) {
                if (map.get(c1) != c2) return false;
            } else {
                if (used.contains(c2)) return false;
                map.put(c1, c2);
                used.add(c2);
            }
        }
        return true; // O(n)
    }

    // 16️⃣ Longest Common Prefix
    public static String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++)
            while (!strs[i].startsWith(prefix))
                prefix = prefix.substring(0, prefix.length() - 1);
        return prefix; // O(n * m)
    }

    // 17️⃣ Reverse Words in a Sentence
    public static String reverseWords(String sentence) {
        String[] words = sentence.trim().split("\\s+");
        Collections.reverse(Arrays.asList(words));
        return String.join(" ", words); // O(n)
    }

    // 18️⃣ Longest Palindromic Substring (Expand Around Center)
    public static String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expand(s, i, i);
            int len2 = expand(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1); // O(n²)
    }

    private static int expand(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    // 19️⃣ Validate if String is a Valid Identifier (like variable name)
    public static boolean isValidIdentifier(String str) {
        return str.matches("[a-zA-Z_$][a-zA-Z\\d_$]*"); // O(n)
    }

    // 20️⃣ Generate All Permutations of a String
    public static void generatePermutations(String str) {
        permute("", str);
    }

    private static void permute(String prefix, String str) {
        if (str.isEmpty())
            System.out.println(prefix);
        else
            for (int i = 0; i < str.length(); i++)
                permute(prefix + str.charAt(i),
                        str.substring(0, i) + str.substring(i + 1)); // O(n!)
    }

    // 🧪 Main Function for Testing
    public static void main(String[] args) {
        System.out.println("Reverse of 'hello': " + reverseString("hello"));
        System.out.println("Is 'madam' palindrome? " + isPalindrome("madam"));
        countVowelsAndConsonants("Interview");
        System.out.println("Are 'listen' and 'silent' anagrams? " + areAnagrams("listen", "silent"));
        System.out.println("First non-repeating in 'aabbcddee': " + firstNonRepeatingChar("aabbcddee"));
        System.out.println("Removed duplicates: " + removeDuplicates("programming"));
        System.out.println("Most frequent char in 'success': " + mostFrequentChar("success"));
        System.out.println("'12345' numeric? " + isNumeric("12345"));
        convertCase("Surya");
        System.out.println("Word count: " + countWords("I love Java programming"));
        System.out.println("Is 'erbottlewat' rotation of 'waterbottle'? " + isRotation("waterbottle", "erbottlewat"));
        System.out.println("Removed whitespaces: " + removeWhitespaces("Java Developer"));
        System.out.println("Compressed: " + compressString("aaabbccaaa"));
        System.out.println("Are 'egg' and 'add' isomorphic? " + areIsomorphic("egg", "add"));
        System.out.println("Longest Common Prefix: " + longestCommonPrefix(new String[]{"flower", "flow", "flight"}));
        System.out.println("Reversed words: " + reverseWords("Java is awesome"));
        System.out.println("Longest palindrome in 'babad': " + longestPalindrome("babad"));
        System.out.println("Valid Identifier 'name_1': " + isValidIdentifier("name_1"));
        System.out.println("\nPermutations of 'abc':");
        generatePermutations("abc");
    }
}
