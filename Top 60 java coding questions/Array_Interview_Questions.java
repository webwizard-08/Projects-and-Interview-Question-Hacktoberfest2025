/*
 * 🧠 Top 20 Array Interview Questions – Java Implementation
 * Author: Sai Surya
 *
 * Description:
 * This program covers 20 essential array-based coding questions frequently asked
 * in technical interviews at companies like FAANG, Infosys, TCS, and Amazon.
 *
 * Each problem includes:
 *  - Problem definition
 *  - Java implementation
 *  - Example input/output
 *  - Time and Space Complexity
 */

import java.util.*;

public class Array_Interview_Questions {

    // 1️⃣ Find the largest element in the array
    public static int findLargest(int[] arr) {
        int max = arr[0];
        for (int num : arr)
            if (num > max)
                max = num;
        return max; // O(n)
    }

    // 2️⃣ Find the smallest element in the array
    public static int findSmallest(int[] arr) {
        int min = arr[0];
        for (int num : arr)
            if (num < min)
                min = num;
        return min; // O(n)
    }

    // 3️⃣ Reverse an array in-place
    public static void reverseArray(int[] arr) {
        int start = 0, end = arr.length - 1;
        while (start < end) {
            int temp = arr[start];
            arr[start++] = arr[end];
            arr[end--] = temp;
        } // O(n)
    }

    // 4️⃣ Find the second largest element
    public static int secondLargest(int[] arr) {
        int first = Integer.MIN_VALUE, second = Integer.MIN_VALUE;
        for (int num : arr) {
            if (num > first) {
                second = first;
                first = num;
            } else if (num > second && num < first) {
                second = num;
            }
        }
        return second; // O(n)
    }

    // 5️⃣ Move all zeros to the end
    public static void moveZeros(int[] arr) {
        int index = 0;
        for (int num : arr)
            if (num != 0)
                arr[index++] = num;
        while (index < arr.length)
            arr[index++] = 0; // O(n)
    }

    // 6️⃣ Find duplicates in an array
    public static void findDuplicates(int[] arr) {
        Set<Integer> seen = new HashSet<>();
        Set<Integer> duplicates = new HashSet<>();
        for (int num : arr)
            if (!seen.add(num))
                duplicates.add(num);
        System.out.println("Duplicates: " + duplicates); // O(n)
    }

    // 7️⃣ Find the missing number from 1 to N
    public static int findMissingNumber(int[] arr, int n) {
        int expectedSum = n * (n + 1) / 2;
        int actualSum = Arrays.stream(arr).sum();
        return expectedSum - actualSum; // O(n)
    }

    // 8️⃣ Kadane’s Algorithm – Maximum Subarray Sum
    public static int maxSubArraySum(int[] arr) {
        int maxSoFar = arr[0], curr = arr[0];
        for (int i = 1; i < arr.length; i++) {
            curr = Math.max(arr[i], curr + arr[i]);
            maxSoFar = Math.max(maxSoFar, curr);
        }
        return maxSoFar; // O(n)
    }

    // 9️⃣ Union of two arrays
    public static Set<Integer> union(int[] a, int[] b) {
        Set<Integer> set = new HashSet<>();
        for (int num : a) set.add(num);
        for (int num : b) set.add(num);
        return set; // O(n+m)
    }

    // 🔟 Intersection of two arrays
    public static Set<Integer> intersection(int[] a, int[] b) {
        Set<Integer> setA = new HashSet<>();
        for (int num : a) setA.add(num);
        Set<Integer> result = new HashSet<>();
        for (int num : b)
            if (setA.contains(num))
                result.add(num);
        return result; // O(n+m)
    }

    // 11️⃣ Sort 0s, 1s, and 2s (Dutch National Flag)
    public static void sort012(int[] arr) {
        int low = 0, mid = 0, high = arr.length - 1;
        while (mid <= high) {
            if (arr[mid] == 0)
                swap(arr, low++, mid++);
            else if (arr[mid] == 1)
                mid++;
            else
                swap(arr, mid, high--);
        } // O(n)
    }

    // 12️⃣ Rotate array by K positions
    public static void rotateArray(int[] arr, int k) {
        k = k % arr.length;
        reverse(arr, 0, arr.length - 1);
        reverse(arr, 0, k - 1);
        reverse(arr, k, arr.length - 1); // O(n)
    }

    // 13️⃣ Find pairs with a given sum
    public static void findPairs(int[] arr, int target) {
        Set<Integer> seen = new HashSet<>();
        for (int num : arr) {
            if (seen.contains(target - num))
                System.out.println("(" + num + ", " + (target - num) + ")");
            seen.add(num);
        } // O(n)
    }

    // 14️⃣ Merge two sorted arrays
    public static int[] mergeSortedArrays(int[] a, int[] b) {
        int i = 0, j = 0, k = 0;
        int[] result = new int[a.length + b.length];
        while (i < a.length && j < b.length)
            result[k++] = (a[i] < b[j]) ? a[i++] : b[j++];
        while (i < a.length) result[k++] = a[i++];
        while (j < b.length) result[k++] = b[j++];
        return result; // O(n+m)
    }

    // 15️⃣ Find majority element (Boyer-Moore)
    public static int majorityElement(int[] arr) {
        int count = 0, candidate = -1;
        for (int num : arr) {
            if (count == 0)
                candidate = num;
            count += (num == candidate) ? 1 : -1;
        }
        return candidate; // O(n)
    }

    // 16️⃣ Find subarray with a given sum (positive integers)
    public static void subarrayWithSum(int[] arr, int sum) {
        int start = 0, curr = 0;
        for (int end = 0; end < arr.length; end++) {
            curr += arr[end];
            while (curr > sum)
                curr -= arr[start++];
            if (curr == sum)
                System.out.println("Subarray found from index " + start + " to " + end);
        } // O(n)
    }

    // 17️⃣ Count frequency of each element
    public static void countFrequency(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr)
            map.put(num, map.getOrDefault(num, 0) + 1);
        System.out.println(map); // O(n)
    }

    // 18️⃣ Find leaders in an array
    public static void findLeaders(int[] arr) {
        int maxRight = arr[arr.length - 1];
        System.out.print(maxRight + " ");
        for (int i = arr.length - 2; i >= 0; i--) {
            if (arr[i] > maxRight) {
                maxRight = arr[i];
                System.out.print(maxRight + " ");
            }
        } // O(n)
    }

    // 19️⃣ Find equilibrium index
    public static int equilibriumIndex(int[] arr) {
        int total = Arrays.stream(arr).sum();
        int leftSum = 0;
        for (int i = 0; i < arr.length; i++) {
            total -= arr[i];
            if (leftSum == total)
                return i;
            leftSum += arr[i];
        }
        return -1; // O(n)
    }

    // 20️⃣ Find missing and repeating numbers
    public static void findMissingAndRepeating(int[] arr) {
        int n = arr.length;
        boolean[] seen = new boolean[n + 1];
        int repeating = -1, missing = -1;
        for (int num : arr) {
            if (seen[num])
                repeating = num;
            else
                seen[num] = true;
        }
        for (int i = 1; i <= n; i++)
            if (!seen[i])
                missing = i;
        System.out.println("Missing: " + missing + ", Repeating: " + repeating);
    }

    // Utility Functions
    private static void reverse(int[] arr, int l, int r) {
        while (l < r) {
            int temp = arr[l];
            arr[l++] = arr[r];
            arr[r--] = temp;
        }
    }

    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // 🧪 Main function to test the problems
    public static void main(String[] args) {
        int[] arr = {1, 3, 5, 7, 3, 0, 2, 1};
        System.out.println("Largest: " + findLargest(arr));
        System.out.println("Smallest: " + findSmallest(arr));

        reverseArray(arr);
        System.out.println("Reversed: " + Arrays.toString(arr));

        System.out.println("Second Largest: " + secondLargest(arr));
        moveZeros(arr);
        System.out.println("Zeros moved: " + Arrays.toString(arr));

        findDuplicates(arr);

        int[] nums = {1, 2, 4, 5};
        System.out.println("Missing Number: " + findMissingNumber(nums, 5));

        int[] kadane = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
        System.out.println("Max Subarray Sum: " + maxSubArraySum(kadane));

        findPairs(new int[]{2, 7, 11, 15}, 9);
    }
}
