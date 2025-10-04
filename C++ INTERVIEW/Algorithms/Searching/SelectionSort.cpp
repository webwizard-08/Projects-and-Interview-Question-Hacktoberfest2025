// Created by: Krasper707
// Date: October 4, 2025
// Description: C++ implementation of the Selection Sort algorithm.

/*
 * <------------------------------------------------------------------->
 *  Selection Sort Algorithm
 * <------------------------------------------------------------------->
 * Time Complexity: O(n^2) -> For all cases (Worst, Average, and Best).
 * Space Complexity: O(1)   -> It's an in-place sorting algorithm.
 *
 * Description: Selection Sort divides the input list into two parts: a sorted
 * sublist of items which is built up from left to right at the front of the
 * list and a sublist of the remaining unsorted items. It proceeds by finding
 * the smallest element in the unsorted sublist, exchanging it with the
 * leftmost unsorted element.
 * -------------------------------------------------------------------
 */

#include <iostream>
#include <vector>
#include <utility>

/**
 * @brief Sorts a vector of integers using the Selection Sort algorithm.
 * @param arr The vector to be sorted.
 */
void selectionSort(std::vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i)
    {
        // Find the minimum element in the unsorted array
        int min_idx = i;
        for (int j = i + 1; j < n; ++j)
        {
            if (arr[j] < arr[min_idx])
            {
                min_idx = j;
            }
        }
        // Swap the found minimum element with the first element
        if (min_idx != i)
        {
            std::swap(arr[min_idx], arr[i]);
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
    std::vector<int> numbers = {64, 25, 12, 22, 11};

    std::cout << "Original array: \n";
    printArray(numbers);

    selectionSort(numbers);

    std::cout << "\nSorted array: \n";
    printArray(numbers);

    return 0;
}