// Program to demonstrate Two-Pointers Technique
#include<bits/stdc++.h>
using namespace std;

// Example 1: Finding a pair with a given sum in a sorted array
// Time Complexity: O(n)
bool pairWithSum(vector<int> &arr, int target){
    int left = 0, right = arr.size() - 1;
    while(left < right){
        int sum = arr[left] + arr[right];
        if(sum == target)
            return true;
        else if(sum < target)
            left++;   // move left pointer forward to increase sum
        else
            right--;  // move right pointer backward to decrease sum
    }
    return false;
}

// Example 2: Reversing an array using two pointers
// Time Complexity: O(n)
void reverseArray(vector<int> &arr){
    int left = 0, right = arr.size() - 1;
    while(left < right){
        swap(arr[left], arr[right]);
        left++;
        right--;
    }
}

int main(){
    vector<int> arr = {1, 2, 3, 4, 5, 6, 7};
    int target = 9;

    cout << "Two-Pointer Example 1: Pair with Sum " << target << " -> "
         << (pairWithSum(arr, target) ? "Found" : "Not Found") << endl;

    cout << "\nOriginal Array: ";
    for(int num : arr) cout << num << " ";

    reverseArray(arr);

    cout << "\nReversed Array: ";
    for(int num : arr) cout << num << " ";

    return 0;
}
