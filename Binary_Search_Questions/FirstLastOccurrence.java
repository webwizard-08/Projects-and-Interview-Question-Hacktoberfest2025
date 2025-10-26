public class FirstLastOccurrence {
    public static int firstOccurrence(int[] a, int target) {
        int l = 0, r = a.length - 1, res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (a[m] == target) { res = m; r = m - 1; }
            else if (a[m] < target) l = m + 1;
            else r = m - 1;
        }
        return res;
    }

    public static int lastOccurrence(int[] a, int target) {
        int l = 0, r = a.length - 1, res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (a[m] == target) { res = m; l = m + 1; }
            else if (a[m] < target) l = m + 1;
            else r = m - 1;
        }
        return res;
    }

    public static void main(String[] args) {
        int[] a = {1,2,2,2,3,4,5};
        int target = 2;
        System.out.println("Array: " + java.util.Arrays.toString(a));
        System.out.println("First occurrence of " + target + ": " + firstOccurrence(a, target)); // expected 1
        System.out.println("Last occurrence of " + target + ": " + lastOccurrence(a, target)); // expected 3
    }
}
