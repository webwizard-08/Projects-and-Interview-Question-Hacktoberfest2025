/*
    Kadane's Algorithm to find maximum subarray sum

    Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

    Time complexity: O(n)
    Space complexity: O(1)
*/

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// Function to find the maximum contiguous subarray sum
int maxSubArraySum(const std::vector<int>& nums) {
    if (nums.empty()) {
        return 0;
    }
    int max_so_far = nums[0];
    int current_max = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        current_max = max(nums[i], current_max + nums[i]);
        max_so_far = max(max_so_far, current_max);
    }
    return max_so_far;
}

int main() {
    vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int max_sum = maxSubArraySum(arr);
    cout << "The input array is: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << std::endl;
    cout << "The maximum subarray sum is: "<<max_sum<<endl;     
    return 0;
}