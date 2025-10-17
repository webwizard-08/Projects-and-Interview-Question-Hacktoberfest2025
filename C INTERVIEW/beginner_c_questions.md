# 15 Beginner C Interview Questions (with Answers)

A collection of fundamental C programming interview questions perfect for beginners. Each question includes a clear explanation and practical examples.

**Difficulty:** Beginner

---

## 1. What is a pointer?

A pointer is a variable that stores the memory address of another variable. Pointers allow direct memory manipulation and are essential for dynamic memory allocation and efficient data passing.

**Example:**
```c
int num = 10;
int *ptr = &num;  // ptr stores the address of num
printf("Value: %d, Address: %p", *ptr, ptr);
```

---

## 2. What is the difference between `malloc` and `calloc`?

Both allocate memory dynamically, but `malloc` allocates uninitialized memory while `calloc` allocates memory and initializes all bytes to zero. `calloc` also takes two parameters (number of elements and size of each element).

**Example:**
```c
int *arr1 = malloc(5 * sizeof(int));  // Uninitialized
int *arr2 = calloc(5, sizeof(int));   // Initialized to 0
```

---

## 3. What is a dangling pointer?

A dangling pointer is a pointer that points to a memory location that has been deallocated or freed. Accessing a dangling pointer leads to undefined behavior and potential crashes.

**Example:**
```c
int *ptr = malloc(sizeof(int));
free(ptr);
// ptr is now dangling - don't use it!
```

---

## 4. How to read/write a file in C?

Use `fopen()` to open a file, `fprintf()`/`fscanf()` for formatted I/O, and `fclose()` to close the file. Always check if the file opened successfully.

**Example:**
```c
FILE *file = fopen("data.txt", "w");
if (file != NULL) {
    fprintf(file, "Hello World");
    fclose(file);
}
```

---

## 5. Explain `struct` vs `union`.

A `struct` allocates memory for all its members separately, while a `union` shares the same memory location for all members. Only one member of a union can be active at a time.

**Example:**
```c
struct Point { int x, y; };        // 8 bytes (2 ints)
union Data { int i; float f; };    // 4 bytes (largest member)
```

---

## 6. What is `const` and `volatile`?

`const` makes a variable read-only and prevents modification. `volatile` tells the compiler that the variable can change unexpectedly (e.g., by hardware) and should not be optimized.

**Example:**
```c
const int MAX_SIZE = 100;     // Cannot be modified
volatile int sensor_data;     // Can change unexpectedly
```

---

## 7. What is a garbage value in C?

A garbage value is an indeterminate value stored in an uninitialized variable. When you declare a variable without initializing it, it contains whatever random data was previously in that memory location. Accessing garbage values leads to unpredictable behavior and bugs.

**Example:**
```c
int num;           // Contains garbage value
printf("%d", num); // Prints unpredictable value
int num2 = 0;      // Properly initialized
printf("%d", num2); // Prints 0
```

---

## 8. How does `sizeof` work?

`sizeof` is a compile-time operator that returns the size in bytes of a variable or data type. It's useful for memory allocation and ensuring portability across different systems.

**Example:**
```c
int arr[10];
printf("Array size: %zu bytes", sizeof(arr));        // 40 bytes
printf("Element size: %zu bytes", sizeof(arr[0]));   // 4 bytes
```

---

## 9. What is pointer arithmetic?

Pointer arithmetic allows you to perform arithmetic operations on pointers. Adding 1 to a pointer moves it to the next element of its type, not just the next byte.

**Example:**
```c
int arr[5] = {1, 2, 3, 4, 5};
int *ptr = arr;
ptr++;  // Now points to arr[1]
printf("%d", *ptr);  // Output: 2
```

---

## 10. What are null-terminated strings and how does `strlen` work?

C strings are arrays of characters terminated by a null character (`'\0'`). `strlen` counts characters until it finds the null terminator, excluding the null character itself.

**Example:**
```c
char str[] = "Hello";
printf("Length: %zu", strlen(str));  // Output: 5
```

---

## 11. Explain the `static` storage class.

`static` variables retain their values between function calls and are initialized only once. Static local variables have local scope but global lifetime, while static global variables have file scope.

**Example:**
```c
void counter() {
    static int count = 0;  // Retains value between calls
    count++;
    printf("Count: %d\n", count);
}
```

---

## 12. How to avoid buffer overflow?

Always check array bounds, use safe functions like `strncpy` instead of `strcpy`, validate input sizes, and use functions that specify buffer sizes to prevent writing beyond allocated memory.

**Example:**
```c
char dest[10];
char src[] = "Very long string";
strncpy(dest, src, sizeof(dest) - 1);  // Safe copy
dest[sizeof(dest) - 1] = '\0';         // Ensure null termination
```

---

## 13. Why use `free()`?

`free()` deallocates memory previously allocated by `malloc()`, `calloc()`, or `realloc()`. It prevents memory leaks by returning memory to the system. Always free dynamically allocated memory when no longer needed.

**Example:**
```c
int *ptr = malloc(sizeof(int));
// Use ptr...
free(ptr);  // Release memory
ptr = NULL; // Good practice: set to NULL
```

---

## 14. What are function pointers? (Simple example)

Function pointers store the address of a function, allowing you to call functions indirectly. They're useful for callbacks, function tables, and dynamic function selection.

**Example:**
```c
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

int (*operation)(int, int) = add;  // Function pointer
printf("Result: %d", operation(5, 3));  // Output: 8
```

---

## 15. Recursion example (factorial)

Recursion is when a function calls itself. The factorial function is a classic example where n! = n × (n-1)! with a base case of 0! = 1.

**Example:**
```c
int factorial(int n) {
    if (n <= 1) return 1;        // Base case
    return n * factorial(n - 1); // Recursive case
}
// factorial(5) = 5 * 4 * 3 * 2 * 1 = 120
```

---

## Summary

These 15 questions cover the fundamental concepts every C programmer should know: pointers, memory management, data types, control structures, and basic programming patterns. Understanding these concepts provides a solid foundation for more advanced C programming topics.
