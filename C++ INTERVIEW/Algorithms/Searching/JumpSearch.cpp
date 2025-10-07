#include <bits/stdc++.h>
using namespace std;

/*
 * JUMP SEARCH ALGORITHM
 * Time Complexity:
 *   - Best Case: O(1) - element found at first jump
 *   - Average Case: O(√n) - optimal jump size is √n
 *   - Worst Case: O(√n) - element at end or not found
 * Space Complexity: O(1) - constant extra space
 * 
 * Jump Search works on sorted arrays by jumping ahead by fixed steps
 * and then performing linear search in the identified block.
 * Optimal jump size is √n for best performance.
 * Better than Linear Search but worse than Binary Search.
 */

// Time Complexity: O(√n) - jumps √n times, then linear search in block
int jumpSearch(vector<int>& arr, int target) {
    int n = arr.size();
    
    // Finding the optimal jump size (√n)
    int jump = sqrt(n);
    int prev = 0;
    
    // Jump through blocks until we find a block where target might exist
    while (arr[min(jump, n) - 1] < target) {
        prev = jump;
        jump += sqrt(n);
        
        // If we've gone beyond the array, element is not present
        if (prev >= n) {
            return -1;
        }
    }
    
    // Linear search in the identified block
    while (arr[prev] < target) {
        prev++;
        
        // If we reach end of current block or array, element not found
        if (prev == min(jump, n)) {
            return -1;
        }
    }
    
    // If element found
    if (arr[prev] == target) {
        return prev;
    }
    
    return -1;  // Element not found
}


int main() {
    // Jump Search requires sorted array
    vector<int> arr = {2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78, 89, 99};
    int target = 23;
    
    cout << "Sorted Array: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
    
    cout << "Jump size (√n): " << sqrt(arr.size()) << endl;
    
    int result = jumpSearch(arr, target);
    
    if (result != -1) {
        cout << "Jump Search: Element " << target << " found at index " << result << endl;
    } else {
        cout << "Jump Search: Element " << target << " not found in the array" << endl;
    }
    
    return 0;
}