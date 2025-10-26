public class BinarySearch {
    public static int binarySearch(int[] a, int target) {
        int l = 0, r = a.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (a[m] == target) return m;
            if (a[m] < target) l = m + 1;
            else r = m - 1;
        }
        return -1;
    }

    public static void main(String[] args) {
        int[] a = {1, 3, 5, 7, 9, 11};
        System.out.println("Array: " + java.util.Arrays.toString(a));
        System.out.println("Index of 7: " + binarySearch(a, 7)); // expected 3
        System.out.println("Index of 2: " + binarySearch(a, 2)); // expected -1
    }
}
