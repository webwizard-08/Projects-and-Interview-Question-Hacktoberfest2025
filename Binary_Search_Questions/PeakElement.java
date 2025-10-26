public class PeakElement {
    // returns index of a peak element (arr[i] > arr[i-1] && arr[i] > arr[i+1])
    public static int findPeak(int[] a) {
        int l = 0, r = a.length - 1;
        while (l < r) {
            int m = l + (r - l) / 2;
            if (a[m] > a[m + 1]) r = m;
            else l = m + 1;
        }
        return l;
    }

    public static void main(String[] args) {
        int[] a = {1,3,20,4,1,0};
        System.out.println("Array: " + java.util.Arrays.toString(a));
        int p = findPeak(a);
        System.out.println("Peak index: " + p + ", value: " + a[p]);
    }
}
