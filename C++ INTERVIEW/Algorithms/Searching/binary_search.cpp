// Created by: Krasper707
// Date: October 4, 2025
// Description: C++ implementation of the Binary Search algorithm.

/*
  -------------------------------------------------------------------
   Binary Search Algorithm
  -------------------------------------------------------------------
  Time Complexity: O(log n) - In each step, we reduce the search space by half.
  Space Complexity: O(1)   - It uses a constant amount of extra space.

  Note: Binary search requires the input array to be sorted.
  -------------------------------------------------------------------
 */

#include <iostream>
#include <vector>
#include <algorithm>

/**
 * @brief Performs a binary search on a sorted vector to find the index of a target element.
 *
 * @param sorted_arr A sorted vector of integers to search within.
 * @param target The integer value to search for.
 * @return The index of the target element if found; otherwise, -1.
 */
int binarySearch(const std::vector<int> &sorted_arr, int target)
{
    int left = 0;
    int right = sorted_arr.size() - 1;

    while (left <= right)
    {
        // Calculate the middle index to avoid potential overflow.
        int mid = left + (right - left) / 2;

        // If the middle element is the target, return its index.
        if (sorted_arr[mid] == target)
        {
            return mid;
        }

        // If the target is greater than the mid element, ignore the left half and look in right half
        if (sorted_arr[mid] < target)
        {
            left = mid + 1;
        }
        // If the target is smaller, ignore the right half and look in the left half.
        else
        {
            right = mid - 1;
        }
    }

    // If the loop finishes and  the target was not found in the array, return -1
    return -1;
}

// --- Main function to demonstrate the Binary Search ---
int main()
{
    // Create a vector of integers.
    // Note: For binary search to work, the array MUST be sorted.
    std::vector<int> numbers = {2, 5, 8, 12, 16, 23, 38, 56, 72, 91};

    // Ex 1: Target is in the array
    int target1 = 23;
    int result1 = binarySearch(numbers, target1);

    if (result1 != -1)
    {
        std::cout << "Target " << target1 << " found at index: " << result1 << std::endl;
    }
    else
    {
        std::cout << "Target " << target1 << " not found." << std::endl;
    }

    // Ex 2: Target is not in the array
    int target2 = 50;
    int result2 = binarySearch(numbers, target2);

    if (result2 != -1)
    {
        std::cout << "Target " << target2 << " found at index: " << result2 << std::endl;
    }
    else
    {
        std::cout << "Target " << target2 << " not found." << std::endl;
    }

    // Ex 3: Target is at the beginning of the array
    int target3 = 2;
    int result3 = binarySearch(numbers, target3);
    std::cout << "Target " << target3 << " found at index: " << result3 << std::endl;

    // Ex 4: Target is at the end of the array
    int target4 = 91;
    int result4 = binarySearch(numbers, target4);
    std::cout << "Target " << target4 << " found at index: " << result4 << std::endl;

    return 0;
}