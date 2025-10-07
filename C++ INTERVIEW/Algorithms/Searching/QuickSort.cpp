#include <bits/stdc++.h>
using namespace std;

/*
 * QUICK SORT ALGORITHM
 * Time Complexity:
 *   - Best Case: O(n log n) - when pivot divides array into equal halves
 *   - Average Case: O(n log n) - when pivot divides array reasonably
 *   - Worst Case: O(n²) - when pivot is always smallest/largest element
 * Space Complexity: O(log n) - for recursion stack in average case
 *                    O(n) - in worst case due to unbalanced recursion
 * 
 * Quick Sort is an in-place, divide-and-conquer sorting algorithm
 * that performs well on average but can degrade to O(n²) in worst case.
 */

// Time Complexity: O(n) - partitions array around pivot
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Time Complexity: O(n log n) average, O(n²) worst case - recursive sorting
void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    vector<int> arr = {10, 7, 8, 9, 1, 5};
    quickSort(arr, 0, arr.size() - 1);

    cout << "Sorted array using Quick Sort: ";
    for (int num : arr) cout << num << " ";
    cout << endl;
    return 0;
}
