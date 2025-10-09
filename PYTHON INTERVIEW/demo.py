def demonstrate_data_types():
    print("=== Python Data Types Demo ===")
    
    # Numbers
    integer_num = 42
    float_num = 3.14
    complex_num = 3 + 4j
    
    print(f"Integer: {integer_num}, Type: {type(integer_num)}")
    print(f"Float: {float_num}, Type: {type(float_num)}")
    print(f"Complex: {complex_num}, Type: {type(complex_num)}")
    
    # Strings
    string_var = "Hello, Python!"
    print(f"String: {string_var}")
    print(f"String length: {len(string_var)}")
    print(f"Uppercase: {string_var.upper()}")
    print(f"Split: {string_var.split(', ')}")
    
    # Lists
    my_list = [1, 2, 3, 4, 5]
    my_list.append(6)
    my_list.insert(0, 0)
    print(f"List: {my_list}")
    print(f"List comprehension: {[x**2 for x in my_list]}")
    
    # Tuples
    my_tuple = (1, 2, 3, 4, 5)
    print(f"Tuple: {my_tuple}")
    print(f"Tuple unpacking: {my_tuple[0]}, {my_tuple[-1]}")
    
    # Dictionaries
    my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
    print(f"Dictionary: {my_dict}")
    print(f"Keys: {list(my_dict.keys())}")
    print(f"Values: {list(my_dict.values())}")
    
    # Sets
    my_set = {1, 2, 3, 4, 5, 5}
    print(f"Set (no duplicates): {my_set}")
    print(f"Set operations: {my_set.union({6, 7, 8})}")

def demonstrate_control_structures():
    print("\n=== Control Structures Demo ===")
    
    # If-else
    age = 20
    if age >= 18:
        print("Adult")
    elif age >= 13:
        print("Teenager")
    else:
        print("Child")
    
    # Loops
    print("For loop with range:")
    for i in range(1, 6):
        print(f"  {i}")
    
    print("While loop:")
    count = 0
    while count < 3:
        print(f"  Count: {count}")
        count += 1
    
    # List comprehension
    squares = [x**2 for x in range(1, 6)]
    print(f"Squares: {squares}")
    
    # Dictionary comprehension
    square_dict = {x: x**2 for x in range(1, 6)}
    print(f"Square dictionary: {square_dict}")

def demonstrate_functions():
    print("\n=== Functions Demo ===")
    
    # Basic function
    def greet(name):
        return f"Hello, {name}!"
    
    print(greet("Alice"))
    
    # Function with default parameters
    def greet_with_title(name, title="Mr."):
        return f"Hello, {title} {name}!"
    
    print(greet_with_title("Smith"))
    print(greet_with_title("Johnson", "Dr."))
    
    # Function with variable arguments
    def sum_all(*args):
        return sum(args)
    
    print(f"Sum of 1,2,3,4,5: {sum_all(1, 2, 3, 4, 5)}")
    
    # Lambda function
    square = lambda x: x**2
    print(f"Square of 5: {square(5)}")
    
    # Higher-order function
    def apply_operation(func, value):
        return func(value)
    
    print(f"Apply square to 4: {apply_operation(square, 4)}")

def demonstrate_oop():
    print("\n=== Object-Oriented Programming Demo ===")
    
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def greet(self):
            return f"Hi, I'm {self.name} and I'm {self.age} years old"
        
        def __str__(self):
            return f"Person(name='{self.name}', age={self.age})"
    
    class Student(Person):
        def __init__(self, name, age, student_id):
            super().__init__(name, age)
            self.student_id = student_id
        
        def study(self):
            return f"{self.name} is studying"
    
    person = Person("Alice", 25)
    print(person.greet())
    print(person)
    
    student = Student("Bob", 20, "S12345")
    print(student.greet())
    print(student.study())
    print(f"Student ID: {student.student_id}")

def demonstrate_exception_handling():
    print("\n=== Exception Handling Demo ===")
    
    def divide_numbers(a, b):
        try:
            result = a / b
            return result
        except ZeroDivisionError:
            return "Cannot divide by zero!"
        except TypeError:
            return "Invalid input types!"
        finally:
            print("Division operation completed")
    
    print(f"10 / 2 = {divide_numbers(10, 2)}")
    print(f"10 / 0 = {divide_numbers(10, 0)}")
    print(f"10 / 'a' = {divide_numbers(10, 'a')}")

def demonstrate_file_operations():
    print("\n=== File Operations Demo ===")
    
    # Write to file
    with open('demo.txt', 'w') as file:
        file.write("Hello, Python!\n")
        file.write("This is a demo file.\n")
    
    print("File 'demo.txt' created successfully")
    
    # Read from file
    with open('demo.txt', 'r') as file:
        content = file.read()
        print("File content:")
        print(content)
    
    # Read line by line
    with open('demo.txt', 'r') as file:
        print("File content line by line:")
        for line_num, line in enumerate(file, 1):
            print(f"Line {line_num}: {line.strip()}")
    
    # Clean up
    import os
    os.remove('demo.txt')
    print("Demo file deleted")

def demonstrate_modules_and_packages():
    print("\n=== Modules and Packages Demo ===")
    
    import math
    import random
    from datetime import datetime
    
    print(f"Square root of 16: {math.sqrt(16)}")
    print(f"Pi value: {math.pi}")
    print(f"Random number: {random.randint(1, 100)}")
    print(f"Current time: {datetime.now()}")

if __name__ == "__main__":
    demonstrate_data_types()
    demonstrate_control_structures()
    demonstrate_functions()
    demonstrate_oop()
    demonstrate_exception_handling()
    demonstrate_file_operations()
    demonstrate_modules_and_packages()
