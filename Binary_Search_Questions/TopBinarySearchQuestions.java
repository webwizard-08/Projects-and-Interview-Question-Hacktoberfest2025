public class TopBinarySearchQuestions {

    // Standard binary search (returns index or -1)
    // Time: O(log n)
    public static int binarySearch(int[] a, int target) {
        int l = 0, r = a.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (a[mid] == target) return mid;
            if (a[mid] < target) l = mid + 1;
            else r = mid - 1;
        }
        return -1;
    }

    // First occurrence of target in sorted array with duplicates
    // Time: O(log n)
    public static int firstOccurrence(int[] a, int target) {
        int l = 0, r = a.length - 1, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (a[mid] == target) { ans = mid; r = mid - 1; }
            else if (a[mid] < target) l = mid + 1;
            else r = mid - 1;
        }
        return ans;
    }

    // Last occurrence of target in sorted array with duplicates
    // Time: O(log n)
    public static int lastOccurrence(int[] a, int target) {
        int l = 0, r = a.length - 1, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (a[mid] == target) { ans = mid; l = mid + 1; }
            else if (a[mid] < target) l = mid + 1;
            else r = mid - 1;
        }
        return ans;
    }

    // Count occurrences using first and last occurrence
    // Time: O(log n)
    public static int countOccurrences(int[] a, int target) {
        int first = firstOccurrence(a, target);
        if (first == -1) return 0;
        int last = lastOccurrence(a, target);
        return last - first + 1;
    }

    // Floor: largest element <= x; returns index or -1 if none
    // Time: O(log n)
    public static int floorIndex(int[] a, int x) {
        int l = 0, r = a.length - 1, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (a[mid] <= x) { ans = mid; l = mid + 1; }
            else r = mid - 1;
        }
        return ans;
    }

    // Ceil: smallest element >= x; returns index or -1 if none
    // Time: O(log n)
    public static int ceilIndex(int[] a, int x) {
        int l = 0, r = a.length - 1, ans = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (a[mid] >= x) { ans = mid; r = mid - 1; }
            else l = mid + 1;
        }
        return ans;
    }

    // Integer sqrt: floor(sqrt(x)) for non-negative x
    // Time: O(log x)
    public static int integerSqrt(int x) {
        if (x < 0) throw new IllegalArgumentException("x must be non-negative");
        if (x < 2) return x;
        int l = 1, r = x / 2, ans = 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            long sq = 1L * mid * mid;
            if (sq == x) return mid;
            if (sq < x) { ans = mid; l = mid + 1; }
            else r = mid - 1;
        }
        return ans;
    }

    // Search in rotated sorted array (no duplicates)
    // Returns index of target or -1
    // Time: O(log n)
    public static int searchInRotated(int[] a, int target) {
        int l = 0, r = a.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (a[mid] == target) return mid;
            // left sorted
            if (a[l] <= a[mid]) {
                if (a[l] <= target && target < a[mid]) r = mid - 1;
                else l = mid + 1;
            } else { // right sorted
                if (a[mid] < target && target <= a[r]) l = mid + 1;
                else r = mid - 1;
            }
        }
        return -1;
    }

    // Peak element: find an index i such that a[i] >= neighbors
    // Works for arrays where adjacent elements may be equal too.
    // Time: O(log n)
    public static int findPeak(int[] a) {
        int n = a.length;
        if (n == 0) return -1;
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (a[mid] > a[mid + 1]) r = mid;
            else l = mid + 1;
        }
        return l; // l == r is a peak
    }

    // Simple main runner with example cases
    public static void main(String[] args) {
        int[] a = {1,2,4,4,4,5,7,9};
        System.out.println("array: java.util.Arrays.toString(a) = " + java.util.Arrays.toString(a));

        System.out.println("binarySearch(5) -> " + binarySearch(a, 5));
        System.out.println("firstOccurrence(4) -> " + firstOccurrence(a, 4));
        System.out.println("lastOccurrence(4) -> " + lastOccurrence(a, 4));
        System.out.println("countOccurrences(4) -> " + countOccurrences(a, 4));

        System.out.println("floorIndex(6) -> " + floorIndex(a, 6) + " (value: " + (floorIndex(a,6) >= 0 ? a[floorIndex(a,6)] : "n/a") + ")");
        System.out.println("ceilIndex(6) -> " + ceilIndex(a, 6) + " (value: " + (ceilIndex(a,6) >= 0 ? a[ceilIndex(a,6)] : "n/a") + ")");

        System.out.println("integerSqrt(26) -> " + integerSqrt(26));

        int[] rotated = { 15, 18, 2, 3, 6, 12 };
        System.out.println("rotated: " + java.util.Arrays.toString(rotated));
        System.out.println("searchInRotated(target=3) -> " + searchInRotated(rotated, 3));

        int[] peaksample = {1,3,20,4,1};
        System.out.println("peaksample: " + java.util.Arrays.toString(peaksample));
        System.out.println("findPeak(peaksample) -> index " + findPeak(peaksample) + " (value: " + peaksample[findPeak(peaksample)] + ")");
    }
}
