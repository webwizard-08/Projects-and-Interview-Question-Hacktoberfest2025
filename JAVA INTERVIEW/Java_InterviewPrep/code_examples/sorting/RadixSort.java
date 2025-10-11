import java.util.Arrays;

/**
 * Implements the Radix Sort algorithm.
 * Radix Sort is a non-comparison based sorting algorithm that sorts integers
 * by processing individual digits. It processes digits from the least
 * significant
 * digit (LSD) to the most significant digit (MSD).
 *
 * Time Complexity: O(d * (n + b)) where d is the number of digits, n is the
 * number of elements, and b is the base (10 for decimal).
 * Space Complexity: O(n + b)
 */
public class RadixSort {

  // A utility function to get the maximum value in arr[]
  private static int getMax(int[] arr) {
    int max = arr[0];
    for (int i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
        max = arr[i];
      }
    }
    return max;
  }

  // A function to do counting sort of arr[] according to the digit represented by
  // exp.
  private static void countingSort(int[] arr, int exp) {
    int n = arr.length;
    int[] output = new int[n]; // output array
    int[] count = new int[10];
    Arrays.fill(count, 0);

    // Store count of occurrences in count[]
    for (int i = 0; i < n; i++) {
      count[(arr[i] / exp) % 10]++;
    }

    // Change count[i] so that count[i] now contains the actual
    // position of this digit in output[]
    for (int i = 1; i < 10; i++) {
      count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = n - 1; i >= 0; i--) {
      output[count[(arr[i] / exp) % 10] - 1] = arr[i];
      count[(arr[i] / exp) % 10]--;
    }

    // Copy the output array to arr[], so that arr[] now
    // contains sorted numbers according to the current digit
    System.arraycopy(output, 0, arr, 0, n);
  }

  /**
   * The main function to that sorts arr[] of size n using Radix Sort.
   * 
   * @param arr The array to be sorted.
   */
  public static void sort(int[] arr) {
    // Find the maximum number to know the number of digits
    int m = getMax(arr);

    // Do counting sort for every digit. Note that instead
    // of passing digit number, exp is passed. exp is 10^i
    // where i is the current digit number.
    for (int exp = 1; m / exp > 0; exp *= 10) {
      countingSort(arr, exp);
    }
  }

  // Main method to test the implementation
  public static void main(String[] args) {
    int[] arr = { 170, 45, 75, 90, 802, 24, 2, 66 };
    System.out.println("Original array: " + Arrays.toString(arr));

    sort(arr);

    System.out.println("Sorted array: " + Arrays.toString(arr));
  }
}