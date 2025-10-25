#include <stdio.h>

void countingSort(int arr[], int n) {
    int output[n];
    int max = arr[0];

    // Find the maximum element
    for (int i = 1; i < n; i++) {
        if (arr[i] > max)
            max = arr[i];
    }

    int count[max + 1];

    // Initialize count array with 0
    for (int i = 0; i <= max; i++)
        count[i] = 0;

    // Store the count of each element
    for (int i = 0; i < n; i++)
        count[arr[i]]++;

    // Change count[i] so that count[i] contains actual position of this element in output array
    for (int i = 1; i <= max; i++)
        count[i] += count[i - 1];

    // Build the output array
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    // Copy the sorted elements into original array
    for (int i = 0; i < n; i++)
        arr[i] = output[i];
}

int main() {
    int arr[] = {4, 2, 2, 8, 3, 3, 1};
    int n = sizeof(arr) / sizeof(arr[0]);

    countingSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    return 0;
}
