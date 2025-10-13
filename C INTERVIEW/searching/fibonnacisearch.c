#include <stdio.h>

// Function to find the minimum of two numbers
int min(int x, int y) {
    return (x <= y) ? x : y;
}

// Fibonacci Search Function
int fibonacciSearch(int arr[], int n, int x) {
    // Initialize the first two Fibonacci numbers
    int fibMMm2 = 0;    // (m-2)'th Fibonacci number
    int fibMMm1 = 1;    // (m-1)'th Fibonacci number
    int fibM = fibMMm2 + fibMMm1;  // m'th Fibonacci number

    // fibM is the smallest Fibonacci number >= n
    while (fibM < n) {
        fibMMm2 = fibMMm1;
        fibMMm1 = fibM;
        fibM = fibMMm2 + fibMMm1;
    }

    // Marks the eliminated range from front
    int offset = -1;

    // While there are elements to be inspected
    while (fibM > 1) {
        // Check if fibMMm2 is a valid location
        int i = min(offset + fibMMm2, n - 1);

        if (arr[i] < x) {
            // Move three Fibonacci variables down
            fibM = fibMMm1;
            fibMMm1 = fibMMm2;
            fibMMm2 = fibM - fibMMm1;
            offset = i;
        }
        else if (arr[i] > x) {
            // Move two Fibonacci variables down
            fibM = fibMMm2;
            fibMMm1 = fibMMm1 - fibMMm2;
            fibMMm2 = fibM - fibMMm1;
        }
        else {
            return i;  // Element found
        }
    }

    // Comparing the last element with x
    if (fibMMm1 && arr[offset + 1] == x) {
        return offset + 1;
    }

    // Element not found
    return -1;
}

int main() {
    int arr[] = {10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 85;  // Element to search

    int index = fibonacciSearch(arr, n, x);

    if (index >= 0)
        printf("Element found at index: %d\n", index);
    else
        printf("Element not found in array.\n");

    return 0;
}
