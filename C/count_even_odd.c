#include <stdio.h>

int main()
{
    int n, i;
    int even = 0, odd = 0;

    // Input array size
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int arr[n];

    // Input array elements
    printf("Enter %d elements:\n", n);
    for (i = 0; i < n; i++)
    {
        scanf("%d", &arr[i]);
    }

    // Count even and odd numbers
    for (i = 0; i < n; i++)
    {
        if (arr[i] % 2 == 0)
            even++;
        else
            odd++;
    }

    // Display results
    printf("Number of even elements = %d\n", even);
    printf("Number of odd elements = %d\n", odd);

    return 0;
}
