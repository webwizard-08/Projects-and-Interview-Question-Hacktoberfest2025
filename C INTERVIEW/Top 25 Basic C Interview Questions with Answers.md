# 🧠 Top 25 Basic C Interview Questions (with Answers)

A beginner-friendly list of commonly asked **C programming interview questions**, focusing on syntax, logic, operators, and data types.

---

## ✅ Basic Concepts

### 1. What is C language?

C is a **procedural programming language** developed by _Dennis Ritchie_ in 1972. It is widely used for system programming and developing operating systems due to its efficiency and performance.

---

### 2. What are the main features of C?

- Simple and efficient
- Portable and platform-independent
- Supports structured programming
- Allows low-level memory access (pointers)
- Has a rich library of built-in functions

---

### 3. What is the structure of a C program?

```c
#include <stdio.h>
int main() {
    // code
    return 0;
}
```

---

### 4. What is a variable?

A variable is a **named memory location** used to store data that can be changed during program execution.

---

### 5. How do you declare a variable in C?

```c
int age;
float salary;
char grade;
```

---

## ✅ Data Types & Operators

### 6. What are the basic data types in C?

- `int`
- `float`
- `char`
- `double`
- `void`

---

### 7. What is the difference between `int` and `float`?

| Type  | Description            | Example    |
| ----- | ---------------------- | ---------- |
| int   | Stores whole numbers   | 5, -10     |
| float | Stores decimal numbers | 3.14, -2.5 |

---

### 8. What are arithmetic operators in C?

`+`, `-`, `*`, `/`, `%`

---

### 9. What is the modulus operator `%` used for?

It returns the **remainder** after division.  
Example:

```c
5 % 2 = 1
```

---

### 10. What is the difference between `++i` and `i++`?

- `++i` → Pre-increment (increments first, then uses the value)
- `i++` → Post-increment (uses the value first, then increments)

---

## ✅ Input / Output

### 11. What is the purpose of `printf()`?

`printf()` is used to display **formatted output** on the screen.  
Example:

```c
printf("Hello World");
```

---

### 12. What is the purpose of `scanf()`?

`scanf()` is used to **take user input**.  
Example:

```c
int num;
scanf("%d", &num);
```

---

### 13. What are format specifiers in C?

They tell the compiler the type of data to read or print.

| Format | Type   |
| ------ | ------ |
| `%d`   | int    |
| `%f`   | float  |
| `%c`   | char   |
| `%s`   | string |

---

## ✅ Conditional Statements

### 14. What is an `if` statement?

It checks a condition and executes a block of code if true.  
Example:

```c
if (a > b) {
    printf("A is greater");
}
```

---

### 15. What is an `if-else` statement?

Executes one block of code if true, otherwise another.  
Example:

```c
if (a > b)
    printf("A is greater");
else
    printf("B is greater");
```

---

### 16. What is a `switch` statement?

Used to execute one block among multiple options based on a variable’s value.  
Example:

```c
switch (choice) {
  case 1: printf("One"); break;
  case 2: printf("Two"); break;
  default: printf("Invalid");
}
```

---

## ✅ Loops & Control Flow

### 17. What is a loop?

A loop allows executing a block of code **repeatedly** until a condition becomes false.

---

### 18. What are the types of loops in C?

- `for` loop
- `while` loop
- `do-while` loop

---

### 19. Write an example of a `for` loop.

```c
for (int i = 1; i <= 5; i++) {
    printf("%d ", i);
}
```

---

### 20. What is the difference between `break` and `continue`?

| Keyword    | Description                                     |
| ---------- | ----------------------------------------------- |
| `break`    | Immediately exits the loop                      |
| `continue` | Skips remaining code and goes to next iteration |

---

## ✅ Arrays & Strings

### 21. What is an array?

An array is a collection of **elements of the same data type** stored in consecutive memory locations.  
Example:

```c
int marks[5] = {90, 85, 70, 80, 95};
```

---

### 22. What is a string in C?

A string is a **sequence of characters** terminated by a null character `'
