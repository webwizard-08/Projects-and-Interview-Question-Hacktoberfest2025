🐍 Top 25 Python Interview Questions
1. What are Python’s key features?

Interpreted, dynamically typed, object-oriented, high-level, and has extensive libraries.

2. What is Python’s difference between list, tuple, set, and dictionary?

List: Mutable, ordered

Tuple: Immutable, ordered

Set: Mutable, unordered, unique elements

Dictionary: Key-value pairs, unordered (Python ≥3.7 insertion ordered)

3. What are Python’s data types?

Numeric (int, float, complex), Sequence (list, tuple, range), Text (str), Set (set, frozenset), Mapping (dict), Boolean (bool), NoneType

4. What is Python’s pass statement?

It’s a null operation, useful as a placeholder.

5. Explain Python’s difference between shallow copy and deep copy.

Shallow copy: Copies references of objects (changes reflect in original if mutable)

Deep copy: Copies objects recursively (independent)

6. What are Python decorators?

Functions that modify other functions or methods without changing their code.

7. What are Python’s mutable and immutable types?

Mutable: list, dict, set

Immutable: int, float, str, tuple, frozenset

8. What is the difference between Python’s is and ==?

is → identity (same object in memory)

== → equality (same value)

9. **What are Python’s *args and kwargs?

*args → variable number of positional arguments

**kwargs → variable number of keyword arguments

10. What are Python’s generators?

Functions that yield values lazily using yield instead of returning all values at once.

11. What is Python’s difference between list comprehension and generator expression?

List comprehension → creates entire list in memory

Generator expression → generates items lazily, memory efficient

12. What are Python’s __init__, __str__, and __repr__ methods?

__init__: constructor

__str__: human-readable string

__repr__: developer-readable string

13. What is Python’s GIL?

Global Interpreter Lock, ensures only one thread executes Python bytecode at a time.

14. Explain Python’s multithreading vs multiprocessing.

Threading: Lightweight, shares memory, affected by GIL

Multiprocessing: Separate memory space, true parallelism

15. What is Python’s difference between @staticmethod, @classmethod, and instance method?

Instance method: Needs object (self)

Class method: Needs class (cls)

Static method: No object/class reference

16. What are Python’s mutable default argument issues?

Using mutable objects (like list/dict) as default arguments can persist changes across function calls.

17. What is Python’s difference between deep copy and shallow copy?

(Covered above, common repeat question)

18. What are Python’s lambda functions?

Anonymous, one-line functions defined using lambda.

19. Explain Python’s map, filter, and reduce.

map(func, iterable) → applies function to each item

filter(func, iterable) → filters items based on condition

reduce(func, iterable) → reduces iterable to a single value (from functools)

20. What are Python’s list slicing methods?
lst[start:stop:step]


Supports negative indexing and skipping elements.

21. What is Python’s difference between append() and extend()?

append() → adds single element

extend() → adds elements from iterable

22. What are Python’s iterators and iterables?

Iterable: Can return an iterator (__iter__)

Iterator: Object with __next__ to traverse elements

23. What is Python’s difference between deepcopy() and copy()?

Already covered in Q5; commonly asked again.

24. What is Python’s difference between range() and xrange()?

Python 2: range() creates list, xrange() is lazy iterator

Python 3: Only range() (lazy)

25. Explain Python’s exception handling.
try:
    # code
except ExceptionType as e:
    # handle
else:
    # runs if no exception
finally:
    # always runs