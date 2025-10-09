#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void demonstratePointers();
void demonstrateArrays();
void demonstrateStrings();
void demonstrateStructures();
void demonstrateDynamicMemory();
void demonstrateFileOperations();

int main() {
    printf("=== C Programming Demo ===\n\n");
    
    demonstratePointers();
    demonstrateArrays();
    demonstrateStrings();
    demonstrateStructures();
    demonstrateDynamicMemory();
    demonstrateFileOperations();
    
    return 0;
}

void demonstratePointers() {
    printf("--- Pointers Demo ---\n");
    
    int num = 42;
    int *ptr = &num;
    
    printf("Value of num: %d\n", num);
    printf("Address of num: %p\n", &num);
    printf("Value of ptr: %p\n", ptr);
    printf("Value pointed by ptr: %d\n", *ptr);
    
    *ptr = 100;
    printf("After changing *ptr, num is now: %d\n\n", num);
}

void demonstrateArrays() {
    printf("--- Arrays Demo ---\n");
    
    int arr[5] = {10, 20, 30, 40, 50};
    int *arr_ptr = arr;
    
    printf("Array elements using array notation:\n");
    for (int i = 0; i < 5; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }
    
    printf("\nArray elements using pointer arithmetic:\n");
    for (int i = 0; i < 5; i++) {
        printf("*(arr_ptr + %d) = %d\n", i, *(arr_ptr + i));
    }
    
    printf("\nSum of array elements: ");
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    printf("%d\n\n", sum);
}

void demonstrateStrings() {
    printf("--- Strings Demo ---\n");
    
    char str1[] = "Hello";
    char str2[] = "World";
    char result[20];
    
    printf("String 1: %s\n", str1);
    printf("String 2: %s\n", str2);
    printf("Length of str1: %zu\n", strlen(str1));
    
    strcpy(result, str1);
    strcat(result, " ");
    strcat(result, str2);
    printf("Concatenated string: %s\n\n", result);
}

struct Student {
    char name[50];
    int age;
    float gpa;
};

void demonstrateStructures() {
    printf("--- Structures Demo ---\n");
    
    struct Student student1;
    strcpy(student1.name, "John Doe");
    student1.age = 20;
    student1.gpa = 3.8;
    
    printf("Student Information:\n");
    printf("Name: %s\n", student1.name);
    printf("Age: %d\n", student1.age);
    printf("GPA: %.2f\n\n", student1.gpa);
}

void demonstrateDynamicMemory() {
    printf("--- Dynamic Memory Allocation Demo ---\n");
    
    int n = 5;
    int *dynamic_arr = (int*)malloc(n * sizeof(int));
    
    if (dynamic_arr == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }
    
    printf("Dynamic array elements: ");
    for (int i = 0; i < n; i++) {
        dynamic_arr[i] = (i + 1) * 10;
        printf("%d ", dynamic_arr[i]);
    }
    printf("\n");
    
    free(dynamic_arr);
    printf("Memory freed successfully\n\n");
}

void demonstrateFileOperations() {
    printf("--- File Operations Demo ---\n");
    
    FILE *file = fopen("demo.txt", "w");
    if (file == NULL) {
        printf("Error opening file for writing\n");
        return;
    }
    
    fprintf(file, "This is a demo file created by C program\n");
    fprintf(file, "Line 2: File operations working correctly\n");
    fclose(file);
    printf("File 'demo.txt' created and written successfully\n");
    
    file = fopen("demo.txt", "r");
    if (file == NULL) {
        printf("Error opening file for reading\n");
        return;
    }
    
    char line[100];
    printf("File contents:\n");
    while (fgets(line, sizeof(line), file) != NULL) {
        printf("%s", line);
    }
    fclose(file);
    
    remove("demo.txt");
    printf("Demo file deleted\n\n");
}
