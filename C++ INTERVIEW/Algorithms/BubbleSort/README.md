# Bubble Sort Algorithm

Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted.

## How it Works

1. **Start** at the beginning of the list.
2. **Compare** the first two elements.
3. **Swap** them if the first is greater than the second.
4. **Move** to the next pair of elements and repeat the comparison and swap.
5. **Continue** this process until the end of the list is reached. This completes one "pass", and the largest element will have "bubbled" up to the end of the list.
6. **Repeat** the passes. For each pass, the next largest element will be placed in its correct position. The number of passes required is one less than the number of elements in the list.

## Time and Space Complexity

- **Best Case Time Complexity:** O(n^2) - This occurs when the list is already sorted.
- **Average Case Time Complexity:** O(n^2) - This is the most common scenario.
- **Worst Case Time Complexity:** O(n^2) - This occurs when the list is sorted in reverse order.
- **Space Complexity:** O(1) - Bubble Sort is an in-place sorting algorithm, meaning it doesn't require any extra space that scales with the input size.
