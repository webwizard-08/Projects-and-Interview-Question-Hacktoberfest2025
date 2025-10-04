// Created by: Krasper707
// Date: October 4, 2025
// Description: C++ implementation of the Insertion Sort algorithm.

/*
 * <------------------------------------------------------------------->
 *  Insertion Sort Algorithm
 * <------------------------------------------------------------------->
 * Time Complexity: O(n^2) -> Worst and Average Case. O(n) -> Best Case (nearly sorted).
 * Space Complexity: O(1)   -> It's an in-place sorting algorithm.
 *
 * Description: Insertion Sort builds the final sorted array one item at a time.
 * It iterates through an input array and removes one element per iteration, finds
 * the place the element belongs in the array, and then places it there.
 * <------------------------------------------------------------------->
 */

#include <iostream>
#include <vector>

/**
 * @brief Sorts a vector of integers using the Insertion Sort algorithm.
 * @param arr The vector to be sorted.
 */
void insertionSort(std::vector<int> &arr)
{
    int n = arr.size();
    for (int i = 1; i < n; ++i)
    {
        int key = arr[i];
        int j = i - 1;

        // Move elements of array arr[0..i-1], that are greater than key,
        // to one position ahead of their current position. Swap them.
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
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
    std::vector<int> numbers = {12, 11, 13, 5, 6};

    std::cout << "Original array: \n";
    printArray(numbers);

    insertionSort(numbers);

    std::cout << "\nSorted array: \n";
    printArray(numbers);

    return 0;
}