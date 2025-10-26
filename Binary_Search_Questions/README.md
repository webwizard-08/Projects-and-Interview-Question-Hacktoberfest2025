Top Binary Search Interview Questions (Java)

This folder contains common binary-search interview problems with Java solutions and a small runner to demonstrate example inputs and outputs.

Included files:

- TopBinarySearchQuestions.java — Several binary-search problems implemented as static methods with example runs in `main`.

Problems covered (with short descriptions):

1. Standard binary search — Find index of a target in a sorted array.
2. First occurrence — Find the first index of the target in a sorted array with duplicates.
3. Last occurrence — Find the last index of the target in a sorted array with duplicates.
4. Count occurrences — Count how many times a target appears in a sorted array (using first/last occurrence).
5. Floor of x — Largest element <= x in a sorted array.
6. Ceil of x — Smallest element >= x in a sorted array.
7. Integer sqrt (floor) — Compute floor(sqrt(x)) using binary search over values.
8. Search in rotated sorted array — Find index of target in an array rotated at unknown pivot.
9. Peak element — Find index of a peak element (nums[i] >= neighbors) using binary search.

How to run:

Open a terminal in this workspace and run (Windows PowerShell):

javac "Binary_Search_Questions\\TopBinarySearchQuestions.java"; java -cp "Binary_Search_Questions" TopBinarySearchQuestions

If Java isn't installed on your machine, install a JDK first (e.g., AdoptOpenJDK / OpenJDK / Oracle JDK).

Notes:
- Each method includes time complexity in comments.
- The `main` demonstrates example usages and prints results.
