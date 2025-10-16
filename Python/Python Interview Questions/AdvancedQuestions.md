1.What is the difference between __str__ and __repr__ in Python?
Answer:

    __str__ returns a user-friendly string representation of the object.

    __repr__ returns an unambiguous string for developers, often usable to recreate the object.


2. What is the difference between is and == in Python?
Answer:

    is checks if two objects reference the same memory location.

    == checks if two objects have the same value.


3. Explain Python’s @property decorator.
Answer:
    It allows a method to be accessed like an attribute, enabling controlled access to private variables.

    class Person:
        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            return self._name


4. What are Python’s args and kwargs?
Answer:

    *args allows passing a variable number of positional arguments.

    **kwargs allows passing a variable number of keyword arguments.


5. Explain Python’s iter() and next().
Answer:

    iter() returns an iterator object from a collection.

    next() retrieves the next item from an iterator; raises StopIteration when finished.


6. What is the difference between Python modules and packages?
Answer:

    A module is a single Python file containing functions or classes.

    A package is a directory containing multiple modules and an __init__.py file.


7. Explain Python’s @staticmethod, @classmethod, and instance methods differences.
Answer:

    Instance methods take self and operate on object instances.

    @classmethod takes cls and operates on the class.

    @staticmethod takes no automatic parameters and behaves like a regular function inside a class.


8. What is a Python iterator?
Answer:
     An iterator is an object that implements the __iter__() and __next__() methods, allowing sequential access to elements in a collection.



9. Explain Python’s zip() function.
Answer: 
    zip() combines multiple iterables element-wise into tuples.

    a = [1, 2]; b = ['x', 'y']
    list(zip(a, b))  # Output: [(1, 'x'), (2, 'y')]


10. What is the difference between Python deepcopy vs pickling/unpickling?
Answer:

    deepcopy creates a duplicate of an object in memory.

    Pickling serializes an object into a byte stream for storage or transfer; unpickling reconstructs it.