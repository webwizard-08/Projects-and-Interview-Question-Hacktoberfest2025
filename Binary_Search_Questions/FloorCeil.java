public class FloorCeil {
    // returns index of floor (largest <= target) or -1 if none
    public static int floorIndex(int[] a, int target) {
        int l = 0, r = a.length - 1, res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (a[m] <= target) { res = m; l = m + 1; }
            else r = m - 1;
        }
        return res;
    }

    // returns index of ceil (smallest >= target) or -1 if none
    public static int ceilIndex(int[] a, int target) {
        int l = 0, r = a.length - 1, res = -1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (a[m] >= target) { res = m; r = m - 1; }
            else l = m + 1;
        }
        return res;
    }

    public static void main(String[] args) {
        int[] a = {1,3,5,7,9};
        System.out.println("Array: " + java.util.Arrays.toString(a));
        System.out.println("Floor index of 6: " + floorIndex(a, 6) + " (value: " + (floorIndex(a,6)>=0?a[floorIndex(a,6)]:"n/a") + ")");
        System.out.println("Ceil index of 6: " + ceilIndex(a, 6) + " (value: " + (ceilIndex(a,6)>=0?a[ceilIndex(a,6)]:"n/a") + ")");
        System.out.println("Floor index of 0: " + floorIndex(a, 0)); // -1
        System.out.println("Ceil index of 10: " + ceilIndex(a, 10)); // -1
    }
}
