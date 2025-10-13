/*
 * 🧠 Top 20 Array Searching Interview Questions – Java Implementation
 * Author: Sai Surya
 *
 * Description:
 * This Java program contains 20 essential array searching problems frequently asked
 * in technical interviews at top companies like FAANG, TCS, Infosys, and Amazon.
 *
 * Each problem includes:
 *  - Problem definition
 *  - Clean Java implementation
 *  - Example I/O
 *  - Time and Space Complexity
 */

import java.util.*;

public class Array_Searching_Questions {

    // 1️⃣ Linear Search
    public static int linearSearch(int[] arr, int key) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i] == key) return i;
        return -1; // O(n)
    }

    // 2️⃣ Binary Search (on sorted array)
    public static int binarySearch(int[] arr, int key) {
        int low = 0, high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == key) return mid;
            else if (arr[mid] < key) low = mid + 1;
            else high = mid - 1;
        }
        return -1; // O(log n)
    }

    // 3️⃣ Search in Rotated Sorted Array
    public static int searchRotatedArray(int[] arr, int key) {
        int low = 0, high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == key) return mid;

            if (arr[low] <= arr[mid]) {
                if (key >= arr[low] && key < arr[mid]) high = mid - 1;
                else low = mid + 1;
            } else {
                if (key > arr[mid] && key <= arr[high]) low = mid + 1;
                else high = mid - 1;
            }
        }
        return -1; // O(log n)
    }

    // 4️⃣ Find First and Last Occurrence
    public static int[] firstLastOccurrence(int[] arr, int key) {
        int first = -1, last = -1;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == key) {
                if (first == -1) first = i;
                last = i;
            }
        }
        return new int[]{first, last}; // O(n)
    }

    // 5️⃣ Search in Nearly Sorted Array
    public static int searchNearlySorted(int[] arr, int key) {
        int low = 0, high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == key) return mid;
            if (mid - 1 >= low && arr[mid - 1] == key) return mid - 1;
            if (mid + 1 <= high && arr[mid + 1] == key) return mid + 1;
            if (arr[mid] > key) high = mid - 2;
            else low = mid + 2;
        }
        return -1; // O(log n)
    }

    // 6️⃣ Find Peak Element
    public static int findPeak(int[] arr) {
        int low = 0, high = arr.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] < arr[mid + 1]) low = mid + 1;
            else high = mid;
        }
        return arr[low]; // O(log n)
    }

    // 7️⃣ Find Minimum in Rotated Sorted Array
    public static int findMinRotated(int[] arr) {
        int low = 0, high = arr.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] > arr[high]) low = mid + 1;
            else high = mid;
        }
        return arr[low]; // O(log n)
    }

    // 8️⃣ Search in 2D Matrix (row-wise & column-wise sorted)
    public static boolean search2DMatrix(int[][] matrix, int key) {
        int row = 0, col = matrix[0].length - 1;
        while (row < matrix.length && col >= 0) {
            if (matrix[row][col] == key) return true;
            else if (matrix[row][col] > key) col--;
            else row++;
        }
        return false; // O(m+n)
    }

    // 9️⃣ Count Occurrences of Element
    public static int countOccurrences(int[] arr, int key) {
        int count = 0;
        for (int num : arr)
            if (num == key) count++;
        return count; // O(n)
    }

    // 🔟 Search in Infinite Sorted Array
    public static int searchInfiniteArray(int[] arr, int key) {
        int low = 0, high = 1;
        while (high < arr.length && arr[high] < key) {
            low = high;
            high *= 2;
        }
        if (high >= arr.length) high = arr.length - 1;
        return binarySearch(Arrays.copyOfRange(arr, low, high + 1), key);
    }

    // 11️⃣ Search in Bitonic Array
    public static int searchBitonicArray(int[] arr, int key) {
        int peak = findPeakIndex(arr);
        int idx = binarySearchAsc(arr, key, 0, peak);
        if (idx != -1) return idx;
        return binarySearchDesc(arr, key, peak + 1, arr.length - 1);
    }

    private static int findPeakIndex(int[] arr) {
        int low = 0, high = arr.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] < arr[mid + 1]) low = mid + 1;
            else high = mid;
        }
        return low;
    }

    private static int binarySearchAsc(int[] arr, int key, int low, int high) {
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == key) return mid;
            else if (arr[mid] < key) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }

    private static int binarySearchDesc(int[] arr, int key, int low, int high) {
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == key) return mid;
            else if (arr[mid] > key) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }

    // 12️⃣ Floor and Ceiling of a Number
    public static int[] floorCeil(int[] arr, int key) {
        int floor = Integer.MIN_VALUE, ceil = Integer.MAX_VALUE;
        for (int num : arr) {
            if (num <= key) floor = Math.max(floor, num);
            if (num >= key) ceil = Math.min(ceil, num);
        }
        return new int[]{floor, ceil}; // O(n)
    }

    // 13️⃣ Search for Pair with Given Sum in Sorted Array
    public static boolean pairWithSum(int[] arr, int sum) {
        int low = 0, high = arr.length - 1;
        while (low < high) {
            int s = arr[low] + arr[high];
            if (s == sum) return true;
            else if (s < sum) low++;
            else high--;
        }
        return false; // O(n)
    }

    // 14️⃣ Search for Triplets with Given Sum
    public static void tripletsWithSum(int[] arr, int sum) {
        Arrays.sort(arr);
        for (int i = 0; i < arr.length - 2; i++) {
            int low = i + 1, high = arr.length - 1;
            while (low < high) {
                int s = arr[i] + arr[low] + arr[high];
                if (s == sum) {
                    System.out.println(arr[i] + "," + arr[low] + "," + arr[high]);
                    low++;
                    high--;
                } else if (s < sum) low++;
                else high--;
            }
        } // O(n²)
    }

    // 15️⃣ Search using Hashing in Unsorted Array
    public static boolean searchUsingHash(int[] arr, int key) {
        Set<Integer> set = new HashSet<>();
        for (int num : arr) set.add(num);
        return set.contains(key); // O(n)
    }

    // 16️⃣ Search in Array with Duplicates (Return all indices)
    public static List<Integer> searchAllIndices(int[] arr, int key) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < arr.length; i++)
            if (arr[i] == key) indices.add(i);
        return indices; // O(n)
    }

    // 17️⃣ Search for Majority Element (Moore’s Voting)
    public static int majorityElement(int[] arr) {
        int count = 0, candidate = -1;
        for (int num : arr) {
            if (count == 0) candidate = num;
            count += (num == candidate) ? 1 : -1;
        }
        return candidate; // O(n)
    }

    // 18️⃣ Find Square Root using Binary Search
    public static int sqrtBinarySearch(int n) {
        int low = 0, high = n, ans = -1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (mid * mid <= n) {
                ans = mid;
                low = mid + 1;
            } else high = mid - 1;
        }
        return ans; // O(log n)
    }

    // 19️⃣ Find Local Minima
    public static int localMinima(int[] arr) {
        int n = arr.length;
        if (n == 1 || arr[0] < arr[1]) return arr[0];
        if (arr[n - 1] < arr[n - 2]) return arr[n - 1];
        for (int i = 1; i < n - 1; i++)
            if (arr[i] < arr[i - 1] && arr[i] < arr[i + 1])
                return arr[i];
        return -1; // O(n)
    }

    // 20️⃣ Search for Subarray with Given Sum (Positive Numbers)
    public static void subarrayWithSum(int[] arr, int sum) {
        int start = 0, currSum = 0;
        for (int end = 0; end < arr.length; end++) {
            currSum += arr[end];
            while (currSum > sum) currSum -= arr[start++];
            if (currSum == sum)
                System.out.println("Subarray from " + start + " to " + end);
        } // O(n)
    }

    // 🧪 Main function to test
    public static void main(String[] args) {
        int[] arr = {1, 3, 5, 7, 9, 11
