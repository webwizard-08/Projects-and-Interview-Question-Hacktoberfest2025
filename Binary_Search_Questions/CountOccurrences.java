public class CountOccurrences {
    public static int count(int[] a, int target) {
        int first = FirstLastOccurrence.firstOccurrence(a, target);
        if (first == -1) return 0;
        int last = FirstLastOccurrence.lastOccurrence(a, target);
        return last - first + 1;
    }

    public static void main(String[] args) {
        int[] a = {1,2,2,2,3,4,5};
        int target = 2;
        System.out.println("Array: " + java.util.Arrays.toString(a));
        System.out.println("Count of " + target + ": " + count(a, target)); // expected 3
        System.out.println("Count of 6: " + count(a, 6)); // expected 0
    }
}
