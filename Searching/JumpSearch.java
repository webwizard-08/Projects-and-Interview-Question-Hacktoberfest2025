// Author: Sai Surya
// Program: Jump Search in Java

public class JumpSearch {
    public static int jumpSearch(int[] arr, int key) {
        int n = arr.length;
        int step = (int) Math.floor(Math.sqrt(n));
        int prev = 0;

        while (arr[Math.min(step, n) - 1] < key) {
            prev = step;
            step += (int) Math.floor(Math.sqrt(n));
            if (prev >= n)
                return -1;
        }

        while (arr[prev] < key) {
            prev++;
            if (prev == Math.min(step, n))
                return -1;
        }

        if (arr[prev] == key)
            return prev;
        return -1;
    }

    public static void main(String[] args) {
        int[] arr = {2, 4, 6, 8, 10, 12, 14, 16};
        int key = 10;

        int result = jumpSearch(arr, key);
        if (result == -1)
            System.out.println("Element not found");
        else
            System.out.println("Element found at index: " + result);
    }
}
