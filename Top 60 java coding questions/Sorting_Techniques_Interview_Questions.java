/*
 * Author: Sai Surya
 *
 * 📘 Description:
 * 
 * This Java program Sorting_Techniques_Interview_Questions.java contains 20 essential
 * sorting-based coding problems frequently asked in technical interviews at
 * top tech companies like FAANG, TCS, Infosys, and Amazon.
 * 
 * Each problem demonstrates different sorting algorithms and variations,
 * from basic comparison sorts to advanced non-comparison sorts, helping learners
 * master both theory and implementation.
 * 
 * 🧩 Topics Covered:
 * - Bubble, Selection, and Insertion Sort
 * - Merge and Quick Sort (Divide & Conquer)
 * - Heap, Counting, Radix, and Bucket Sort
 * - Custom Comparator and Object Sorting
 * - Sorting by frequency, keys, and multiple attributes
 * - Lexicographical, Descending, and Numeric sorting
 * 
 * 💡 Each Problem Includes:
 * - Problem definition
 * - Java implementation
 * - Example I/O
 * - Time & Space Complexity
 * 
 * 🎯 Usage:
 * - Strengthen sorting concepts
 * - Prepare for coding interviews
 * - Understand when and why to use each algorithm
 * - Improve analytical thinking for DSA problems
 */

import java.util.*;

public class Sorting_Techniques_Interview_Questions {

    // 1. Bubble Sort
    public static void bubbleSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = 0; j < arr.length - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }

    // 2. Selection Sort
    public static void selectionSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[j] < arr[minIdx]) minIdx = j;
            }
            int temp = arr[i];
            arr[i] = arr[minIdx];
            arr[minIdx] = temp;
        }
    }

    // 3. Insertion Sort
    public static void insertionSort(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    // 4. Merge Sort
    public static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = (left + right) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }

    private static void merge(int[] arr, int left, int mid, int right) {
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        while (i <= mid && j <= right) {
            temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
        }
        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];
        for (i = left; i <= right; i++) arr[i] = temp[i - left];
    }

    // 5. Quick Sort
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = (low - 1);
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }

    // 6. Heap Sort
    public static void heapSort(int[] arr) {
        int n = arr.length;
        for (int i = n / 2 - 1; i >= 0; i--) heapify(arr, n, i);
        for (int i = n - 1; i > 0; i--) {
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
            heapify(arr, i, 0);
        }
    }

    private static void heapify(int[] arr, int n, int i) {
        int largest = i, left = 2 * i + 1, right = 2 * i + 2;
        if (left < n && arr[left] > arr[largest]) largest = left;
        if (right < n && arr[right] > arr[largest]) largest = right;
        if (largest != i) {
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;
            heapify(arr, n, largest);
        }
    }

    // 7. Counting Sort
    public static void countingSort(int[] arr) {
        int max = Arrays.stream(arr).max().getAsInt();
        int[] count = new int[max + 1];
        for (int num : arr) count[num]++;
        int idx = 0;
        for (int i = 0; i <= max; i++) {
            while (count[i]-- > 0) arr[idx++] = i;
        }
    }

    // 8. Radix Sort
    public static void radixSort(int[] arr) {
        int max = Arrays.stream(arr).max().getAsInt();
        for (int exp = 1; max / exp > 0; exp *= 10) countingSortByDigit(arr, exp);
    }

    private static void countingSortByDigit(int[] arr, int exp) {
        int n = arr.length;
        int[] output = new int[n];
        int[] count = new int[10];
        for (int i = 0; i < n; i++) count[(arr[i] / exp) % 10]++;
        for (int i = 1; i < 10; i++) count[i] += count[i - 1];
        for (int i = n - 1; i >= 0; i--) {
            output[count[(arr[i] / exp) % 10] - 1] = arr[i];
            count[(arr[i] / exp) % 10]--;
        }
        System.arraycopy(output, 0, arr, 0, n);
    }

    // 9. Bucket Sort
    public static void bucketSort(float[] arr) {
        int n = arr.length;
        if (n <= 0) return;
        @SuppressWarnings("unchecked")
        Vector<Float>[] buckets = new Vector[n];
        for (int i = 0; i < n; i++) buckets[i] = new Vector<>();
        for (float v : arr) {
            int idx = (int) v * n;
            buckets[idx].add(v);
        }
        for (Vector<Float> bucket : buckets) Collections.sort(bucket);
        int index = 0;
        for (Vector<Float> bucket : buckets)
            for (float val : bucket) arr[index++] = val;
    }

    // 10. Sort by Frequency
    public static void sortByFrequency(int[] arr) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : arr) freq.put(num, freq.getOrDefault(num, 0) + 1);
        Arrays.sort(arr, 0, arr.length, (a, b) -> {
            int f1 = freq.get(a), f2 = freq.get(b);
            return f1 == f2 ? a - b : f2 - f1;
        });
    }

    // 11. Sort Strings Lexicographically
    public static void sortStrings(String[] arr) {
        Arrays.sort(arr);
    }

    // 12. Sort 0s, 1s, and 2s (Dutch National Flag)
    public static void sort012(int[] arr) {
        int low = 0, mid = 0, high = arr.length - 1;
        while (mid <= high) {
            if (arr[mid] == 0) {
                int temp = arr[low];
                arr[low++] = arr[mid];
                arr[mid++] = temp;
            } else if (arr[mid] == 1) mid++;
            else {
                int temp = arr[mid];
                arr[mid] = arr[high];
                arr[high--] = temp;
            }
        }
    }

    // 13. Sort by Absolute Difference
    public static void sortByAbsDiff(int[] arr, int k) {
        Arrays.sort(arr, 0, arr.length, Comparator.comparingInt(a -> Math.abs(a - k)));
    }

    // 14. Sort Objects Using Comparator
    static class Student {
        String name;
        int marks;
        Student(String name, int marks) {
            this.name = name;
            this.marks = marks;
        }
    }

    public static void sortStudentsByMarks(Student[] students) {
        Arrays.sort(students, Comparator.comparingInt(s -> s.marks));
    }

    // 15. Sort in Descending Order
    public static void sortDescending(int[] arr) {
        Integer[] temp = Arrays.stream(arr).boxed().toArray(Integer[]::new);
        Arrays.sort(temp, Collections.reverseOrder());
        for (int i = 0; i < arr.length; i++) arr[i] = temp[i];
    }

    // 16. Find Kth Smallest Element
    public static int kthSmallest(int[] arr, int k) {
        Arrays.sort(arr);
        return arr[k - 1];
    }

    // 17. Sort Characters by Frequency
    public static String sortCharsByFrequency(String s) {
        Map<Character, Integer> freq = new HashMap<>();
        for (char c : s.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);
        List<Character> chars = new ArrayList<>(freq.keySet());
        chars.sort((a, b) -> freq.get(b) - freq.get(a));
        StringBuilder sb = new StringBuilder();
        for (char c : chars) sb.append(String.valueOf(c).repeat(freq.get(c)));
        return sb.toString();
    }

    // 18. Sort Names Alphabetically
    public static void sortNames(String[] names) {
        Arrays.sort(names, String::compareToIgnoreCase);
    }

    // 19. Sort 2D Matrix Row-wise
    public static void sortMatrix(int[][] mat) {
        for (int[] row : mat) Arrays.sort(row);
    }

    // 20. Compare Sorting Algorithms (Print Time Complexity)
    public static void compareSortingAlgorithms() {
        System.out.println("Bubble Sort: O(n^2)");
        System.out.println("Insertion Sort: O(n^2)");
        System.out.println("Merge Sort: O(n log n)");
        System.out.println("Quick Sort: O(n log n)");
        System.out.println("Heap Sort: O(n log n)");
        System.out.println("Counting Sort: O(n + k)");
        System.out.println("Radix Sort: O(nk)");
        System.out.println("Bucket Sort: O(n + k)");
    }

    // Example main method
    public static void main(String[] args) {
        int[] arr = {5, 1, 4, 2, 8};
        bubbleSort(arr);
        System.out.println("Bubble Sorted: " + Arrays.toString(arr));

        int[] quickArr = {9, 4, 6, 2, 10, 3};
        quickSort(quickArr, 0, quickArr.length - 1);
        System.out.println("Quick Sorted: " + Arrays.toString(quickArr));
    }
}
