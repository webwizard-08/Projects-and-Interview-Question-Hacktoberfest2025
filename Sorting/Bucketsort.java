import java.util.*;

public class BucketSort {
    public static void bucketSort(float[] arr) {
        int n = arr.length;
        if (n <= 0) return;
        List<Float>[] buckets = new List[n];
        for (int i = 0; i < n; i++) buckets[i] = new ArrayList<>();
        for (float num : arr) {
            int index = (int)(num * n);
            buckets[index].add(num);
        }
        for (List<Float> bucket : buckets) Collections.sort(bucket);
        int idx = 0;
        for (List<Float> bucket : buckets)
            for (float num : bucket)
                arr[idx++] = num;
    }

    public static void main(String[] args) {
        float[] arr = {0.42f, 0.32f, 0.23f, 0.52f, 0.25f, 0.47f, 0.51f};
        bucketSort(arr);
        for (float num : arr) System.out.print(num + " ");
    }
}
