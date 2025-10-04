// Created by: Krasper707
// Date: October 4, 2025
// Description: C++ implementation of the Bubble Sort algorithm.

/*
 * <------------------------------------------------------------------->
 *  Bubble Sort Algorithm
 * <------------------------------------------------------------------->
 * Time Complexity: O(n^2) -> Worst and Average Case. O(n) -> Best Case (if already sorted).
 * Space Complexity: O(1)   -> It's an in-place sorting algorithm.
 *
 * Description: Bubble Sort repeatedly steps through the list, compares adjacent
 * elements and swaps them if they are in the wrong order. The pass through
 * the list is repeated until the list is sorted.
 * <------------------------------------------------------------------->
 */

#include <iostream>
#include <vector>
#include <utility>

/**
 * @brief Sorts a vector of integers using the Bubble Sort algorithm.
 * @param arr The vector to be sorted.
 */
void bubbleSort(std::vector<int> &arr)
{
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n - 1; ++i)
    {
        swapped = false;
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        // If no two elements were swapped by inner loop, then the array is sorted and we can break out from this loop
        if (!swapped)
        {
            break;
        }
    }
}

/**
 * @brief Utility function to print the elements of a vector.
 * @param arr The vector to print.
 */
void printArray(const std::vector<int> &arr)
{
    for (int num : arr)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main()
{
    std::vector<int> numbers = {64, 34, 25, 12, 22, 11, 90};

    std::cout << "Original array: \n";
    printArray(numbers);

    bubbleSort(numbers);

    std::cout << "\nSorted array: \n";
    printArray(numbers);

    return 0;
}