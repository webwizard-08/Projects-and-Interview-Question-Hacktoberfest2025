# Top 30 Java Interview Questions (with Answers)

A comprehensive list of commonly asked Java interview questions covering core concepts, OOP, collections, exceptions, multithreading, and Java 8+ features.

---

## ✅ Core Java & OOP Concepts

1. **What are the main features of Java?**  
   Platform-independent, Object-oriented, Secure, Robust, Multithreaded, High-performance (JIT compiler).

2. **What is the difference between JDK, JRE, and JVM?**  
   - **JVM**: Executes Java bytecode.  
   - **JRE**: JVM + runtime libraries.  
   - **JDK**: JRE + development tools.

3. **What is the difference between an object and a class?**  
   Class is a blueprint; object is an instance of a class.

4. **What are constructors in Java?**  
   Special methods used to initialize objects. They have the same name as the class and no return type.

5. **What is method overloading and overriding?**  
   - **Overloading**: Same method name, different parameters (compile-time).  
   - **Overriding**: Subclass redefines superclass method (runtime).

---

## ✅ Inheritance, Polymorphism & Abstraction

6. **What is inheritance in Java?**  
   Mechanism where one class inherits properties of another (`extends` keyword).

7. **What is polymorphism?**  
   Ability of a variable, function, or object to take many forms (compile-time & runtime).

8. **What is abstraction?**  
   Hiding internal implementation and showing only functionality (via abstract classes or interfaces).

9. **What is the difference between abstract class and interface?**  
   - Abstract class: Partial abstraction, can have state.  
   - Interface: 100% abstraction (before Java 8), multiple inheritance, no state (until Java 8 default methods).

10. **Can Java support multiple inheritance?**  
    Not with classes, but supported via interfaces.

---

## ✅ Encapsulation, Access Modifiers

11. **What is encapsulation?**  
    Wrapping data (variables) and code (methods) into a single unit. Achieved using private variables and public getters/setters.

12. **What are access modifiers in Java?**  
    - `private`, `default` (package-private), `protected`, `public`.

---

## ✅ Java Collections Framework (JCF)

13. **What is the difference between List, Set, and Map?**  
    - **List**: Ordered, allows duplicates.  
    - **Set**: Unordered, no duplicates.  
    - **Map**: Key-value pairs, keys are unique.

14. **Difference between ArrayList and LinkedList?**  
    - ArrayList: Fast for read, slow for insert/delete.  
    - LinkedList: Fast for insert/delete, slow for read.

15. **Difference between HashMap and TreeMap?**  
    - HashMap: Unordered, faster, allows null key.  
    - TreeMap: Sorted by keys, slower, no null key (throws `NullPointerException`).

16. **What is the difference between HashSet and HashMap?**  
    - HashSet stores only values; HashMap stores key-value pairs.

17. **What are the fail-fast and fail-safe iterators?**  
    - Fail-fast: Throws `ConcurrentModificationException` (e.g., `ArrayList`, `HashMap`).  
    - Fail-safe: Does not throw (e.g., `ConcurrentHashMap`, `CopyOnWriteArrayList`).

---

## ✅ Exception Handling

18. **What is the difference between checked and unchecked exceptions?**  
    - Checked: Must be handled (e.g., `IOException`).  
    - Unchecked: Runtime exceptions (e.g., `NullPointerException`).

19. **What is finally block?**  
    Executes always after try/catch, used for cleanup.

20. **Can a finally block be skipped?**  
    Only if `System.exit()` is called or JVM crashes.

---

## ✅ Multithreading & Concurrency

21. **What is the difference between `synchronized` method and block?**  
    - Method: Locks entire method.  
    - Block: Locks only specific code inside method.

22. **What is the difference between `Thread` and `Runnable`?**  
    - `Thread`: Subclass to create thread.  
    - `Runnable`: Interface, preferred (supports multiple inheritance).

23. **What is the difference between `wait()` and `sleep()`?**  
    - `wait()`: Releases the lock, used for inter-thread communication.  
    - `sleep()`: Holds lock, just pauses thread execution.

24. **What is a deadlock?**  
    When two threads wait indefinitely for each other's resources.

---

## ✅ Java 8+ Features

25. **What are Lambda expressions?**  
    A concise way to represent functional interfaces (`(a, b) -> a + b`).

26. **What are functional interfaces?**  
    Interface with a single abstract method (e.g., `Runnable`, `Comparator`, `Function<T, R>`).

27. **What are Streams in Java 8?**  
    Used to process collections in a functional style (`filter`, `map`, `collect`).

28. **Difference between `map()` and `flatMap()` in streams?**  
    - `map()`: Transforms each element.  
    - `flatMap()`: Flattens nested structures.

---

## ✅ Memory Management & Internals

29. **What is garbage collection in Java?**  
    Automatic memory management that removes unreferenced objects.

30. **What are the different memory areas in JVM?**  
    - **Heap**: Objects.  
    - **Stack**: Method calls, local vars.  
    - **Method Area**: Class metadata.  
    - **Program Counter**: Current execution line.  
    - **Native Method Stack**: Native code.

---

> 💡 *Tip: Practice these questions and try to write code snippets or real-world use cases to solidify understanding.*
