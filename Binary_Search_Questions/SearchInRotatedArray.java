public class SearchInRotatedArray {
    public static int search(int[] a, int target) {
        int l = 0, r = a.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (a[m] == target) return m;
            // left half sorted
            if (a[l] <= a[m]) {
                if (a[l] <= target && target < a[m]) r = m - 1;
                else l = m + 1;
            } else { // right half sorted
                if (a[m] < target && target <= a[r]) l = m + 1;
                else r = m - 1;
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        int[] a = {4,5,6,7,0,1,2};
        System.out.println("Array: " + java.util.Arrays.toString(a));
        System.out.println("Index of 0: " + search(a, 0)); // expected 4
        System.out.println("Index of 5: " + search(a, 5)); // expected 1
        System.out.println("Index of 3: " + search(a, 3)); // expected -1
    }
}
