#include <stdio.h>

int main()
{
    int n, i;
    int sum = 0;

    // Take array size input
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int arr[n]; // Declare array

    // Take array elements input
    printf("Enter %d elements:\n", n);
    for (i = 0; i < n; i++)
    {
        scanf("%d", &arr[i]);
        sum += arr[i]; // Add to sum
    }

    // Display the sum
    printf("Sum of array elements = %d\n", sum);

    return 0;
}
