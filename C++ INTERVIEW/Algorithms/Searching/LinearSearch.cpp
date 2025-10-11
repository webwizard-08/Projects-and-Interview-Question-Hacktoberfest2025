#include <bits/stdc++.h>
using namespace std;

/*
 * LINEAR SEARCH ALGORITHM
 * Time Complexity:
 *   - Best Case: O(1) - element found at first position
 *   - Average Case: O(n) - element found at middle position
 *   - Worst Case: O(n) - element found at last position or not found
 * Space Complexity: O(1) - constant extra space
 * 
 * Linear Search is the simplest searching algorithm that checks
 * every element sequentially until the target is found or array ends.
 * Works on both sorted and unsorted arrays.
 */

// Time Complexity: O(n) - may need to check all elements
int linearSearch(vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return i;  // Return index if element found
        }
    }
    return -1;  // Return -1 if element not found
}


int main() {
    vector<int> arr = {64, 25, 12, 22, 11, 90, 5, 77, 30};
    int target = 22;
    
    cout << "Array: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
    
    int result = linearSearch(arr, target);
    
    if (result != -1) {
        cout << "Linear Search: Element " << target << " found at index " << result << endl;
    } else {
        cout << "Linear Search: Element " << target << " not found in the array" << endl;
    }
    
    return 0;
}