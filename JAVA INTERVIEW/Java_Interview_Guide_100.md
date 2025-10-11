# Comprehensive Java Interview Guide (100 Q&A)

A complete list of 100 essential Java interview questions and answers, covering everything from core concepts to advanced topics.

---

##  Core Java Fundamentals

**Q1: What are the main features of Java?**

**A:** Java is platform-independent (Write Once, Run Anywhere), object-oriented, secure, robust, and multithreaded. It uses a Just-In-Time (JIT) compiler for high performance.

---

**Q2: What is the difference between JDK, JRE, and JVM?**

**A:**
-   **JVM (Java Virtual Machine):** An abstract machine that executes Java bytecode.
-   **JRE (Java Runtime Environment):** The environment needed to *run* Java applications. It contains the JVM and runtime libraries.
-   **JDK (Java Development Kit):** The full toolkit for *developing* Java applications. It contains the JRE plus development tools like the compiler (`javac`).

---

**Q3: What are the 8 primitive data types in Java?**

**A:** `byte`, `short`, `int`, `long`, `float`, `double`, `char`, and `boolean`.

---

**Q4: What is the difference between `==` and the `.equals()` method?**

**A:**
-   `==`: Checks if two references point to the **same memory location**.
-   `.equals()`: Checks if the two objects have the **same value** or content. This method should be overridden in custom classes.

---

**Q5: Why is the `String` class immutable in Java?**

**A:** Once a `String` object is created, its value cannot be changed. This is for security, thread safety, and performance (allowing for caching in the String Pool).

---

**Q6: What is the String Pool?**

**A:** A special storage area in the Java heap where String literals are stored. If you create a new String literal, the JVM checks the pool first and returns a reference to an existing string if it finds a match, saving memory.

---

**Q7: What is the difference between `String`, `StringBuilder`, and `StringBuffer`?**

**A:**
-   **`String`**: Immutable. A new object is created for every modification.
-   **`StringBuilder`**: Mutable and not thread-safe. It's fast and should be used for string manipulation in a single thread.
-   **`StringBuffer`**: Mutable and thread-safe (its methods are synchronized). It's slower and should be used when multiple threads might modify the string.

---

**Q8: What are Autoboxing and Unboxing?**

**A:**
-   **Autoboxing**: The automatic conversion of a primitive type to its corresponding wrapper class object (e.g., `int` to `Integer`).
-   **Unboxing**: The automatic conversion of a wrapper class object back to its primitive type (e.g., `Integer` to `int`).

---

**Q9: What is the `static` keyword?**

**A:** The `static` keyword indicates that a member (variable or method) belongs to the **class itself**, not to an instance of the class. All instances share the same static variable, and static methods can be called without creating an object.

---

**Q10: What are the `final`, `finally`, and `finalize` keywords?**

**A:**
-   **`final`**: A modifier to make a variable a constant, a method non-overridable, or a class non-inheritable.
-   **`finally`**: A block in a `try-catch` statement that **always executes**. It's used for cleanup.
-   **`finalize`**: A method called by the garbage collector just before an object is destroyed. Its use is highly discouraged.

---

##  Object-Oriented Programming (OOP)

**Q11: What are the four main principles of OOP?**

**A:** Encapsulation, Inheritance, Polymorphism, and Abstraction.

---

**Q12: What is the difference between an object and a class?**

**A:** A **class** is a blueprint or template for creating objects. An **object** is a runnable instance of a class.

---

**Q13: What are constructors in Java?**

**A:** Special methods used to initialize objects. They have the same name as the class and no return type.

---

**Q14: What is Encapsulation?**

**A:** Wrapping data (variables) and the code that acts on it (methods) into a single unit (a class). It's achieved by making variables `private` and providing `public` getter and setter methods.

---

**Q15: What is Inheritance?**

**A:** A mechanism where one class (the subclass) inherits the properties and methods of another class (the superclass) using the `extends` keyword.

---

**Q16: What is Polymorphism?**

**A:** The ability of an object to take on many forms. In Java, it's achieved through method **overloading** (compile-time) and method **overriding** (runtime).

---

**Q17: What is method overloading vs. overriding?**

**A:**
-   **Overloading**: Multiple methods in the same class with the same name but different parameters.
-   **Overriding**: A subclass provides a specific implementation for a method that is already defined in its superclass.

---

**Q18: What is Abstraction?**

**A:** Hiding the complex implementation details and showing only the essential features of an object. It's achieved using **abstract classes** and **interfaces**.

---

**Q19: What is the difference between an abstract class and an interface?**

**A:**
-   **Abstract Class**: Can have both abstract and non-abstract methods. A class can extend only one abstract class.
-   **Interface**: Can only have abstract methods (before Java 8). A class can implement multiple interfaces.

---

**Q20: Can Java support multiple inheritance?**

**A:** Not with classes. However, a class can implement multiple interfaces, which allows for a form of multiple inheritance of type.

---

**Q21: What is the `super` keyword?**

**A:** A reference to the immediate parent class, used to call the superclass's constructor (`super()`) or to access its members.

---

**Q22: What is an immutable object?**

**A:** An object whose state cannot be changed after it is created (e.g., `String`, `Integer`).

---

##  Java Collections Framework (JCF)

**Q23: What is the root interface of the Collections hierarchy?**

**A:** The `Collection` interface. (`Map` is separate and does not extend `Collection`).

---

**Q24: What is the difference between `List`, `Set`, and `Map`?**

**A:**
-   **`List`**: An ordered collection that allows duplicate elements.
-   **`Set`**: An unordered collection that does not allow duplicate elements.
-   **`Map`**: A collection of key-value pairs where keys must be unique.

---

**Q25: What is the difference between `ArrayList` and `LinkedList`?**

**A:**
-   **`ArrayList`**: Backed by a dynamic array. Fast for random access (`get()`), slow for insertion/deletion.
-   **`LinkedList`**: Backed by a doubly linked list. Fast for insertion/deletion, slow for random access.

---

**Q26: What is the difference between `HashMap` and `TreeMap`?**

**A:**
-   **`HashMap`**: Unordered, uses hashing. Faster (O(1) average) and allows one `null` key.
-   **`TreeMap`**: Sorted by keys. Slower (O(log n)) and does not allow `null` keys.

---

**Q27: What is the difference between `HashSet` and `TreeSet`?**

**A:**
-   **`HashSet`**: Unordered, uses a `HashMap` internally. It's faster.
-   **`TreeSet`**: Sorted, uses a `TreeMap` internally. It's slower.

---

**Q28: How does a `HashMap` work internally?**

**A:** It uses an array of buckets. When you put an element, it calculates the key's `hashCode()`, finds a bucket index, and stores the entry. Collisions are handled by storing entries in a linked list (or a tree in Java 8+) within the same bucket.

---

**Q29: What is the contract between `equals()` and `hashCode()`?**

**A:** If two objects are equal according to `.equals()`, they **must** have the same hash code. If two objects have the same hash code, they are not necessarily equal.

---

**Q30: What is the difference between `Comparable` and `Comparator`?**

**A:**
-   **`Comparable`**: Provides a single, "natural" ordering for a class (the class implements it).
-   **`Comparator`**: Provides an external, alternative way to sort objects (a separate class implements it).

---

**Q31: What is the difference between `Iterator` and `ListIterator`?**

**A:**
-   **`Iterator`**: Allows forward traversal only.
-   **`ListIterator`**: Allows bidirectional (forward and backward) traversal and element modification.

---

**Q32: What are fail-fast and fail-safe iterators?**

**A:**
-   **Fail-fast**: Throws a `ConcurrentModificationException` if the collection is modified while iterating (e.g., `ArrayList`).
-   **Fail-safe**: Does not throw an exception because it operates on a clone of the collection (e.g., `ConcurrentHashMap`).

---

**Q33: What is `ConcurrentHashMap`?**

**A:** A thread-safe version of `HashMap` that provides higher concurrency by locking only segments of the map.

---

**Q34: What is the difference between `ArrayList` and `Vector`?**

**A:** `Vector` is an older, synchronized (thread-safe) version of `ArrayList`. `ArrayList` is not synchronized and is generally preferred for better performance.

---

##  Exception Handling

**Q35: What is the base class for all exceptions in Java?**

**A:** `java.lang.Throwable`.

---

**Q36: What is the difference between checked and unchecked exceptions?**

**A:**
-   **Checked Exceptions**: Must be declared or handled at compile time (e.g., `IOException`).
-   **Unchecked Exceptions**: Do not need to be handled at compile time (e.g., `NullPointerException`).

---

**Q37: What is the `finally` block?**

**A:** A block that always executes after a `try-catch` statement, used for cleanup.

---

**Q38: Can a `finally` block be skipped?**

**A:** Yes, only if `System.exit()` is called or if the JVM crashes.

---

**Q39: What is the `try-with-resources` statement?**

**A:** A `try` block that automatically closes resources that implement the `AutoCloseable` interface, eliminating the need for a `finally` block.

---

**Q40: What is the difference between `throw` and `throws`?**

**A:**
-   **`throw`**: A keyword used to manually **throw an instance** of an exception.
-   **`throws`**: A keyword in a method's signature to **declare that the method might throw** an exception.

---

**Q41: Can you create a custom exception?**

**A:** Yes, by creating a class that extends `Exception` or `RuntimeException`.

---

##  Multithreading & Concurrency

**Q42: What are the two ways to create a thread in Java?**

**A:** By extending the `Thread` class or by implementing the `Runnable` interface. Implementing `Runnable` is preferred.

---

**Q43: What is the lifecycle of a thread?**

**A:** New, Runnable, Blocked, Waiting, Timed_Waiting, and Terminated.

---

**Q44: What is the difference between `wait()` and `sleep()`?**

**A:**
-   **`wait()`**: **Releases the lock** and is used for inter-thread communication.
-   **`sleep()`**: **Does not release the lock** and simply pauses the thread for a specified time.

---

**Q45: What is the difference between a `synchronized` method and a `synchronized` block?**

**A:** A `synchronized` method locks the entire object. A `synchronized` block locks only the object specified in the parentheses, allowing for more granular control.

---

**Q46: What is the `volatile` keyword?**

**A:** It guarantees that any read of a `volatile` variable will see the most recent write by any thread. It ensures **visibility** but not atomicity.

---

**Q47: What is a deadlock?**

**A:** A situation where two or more threads are blocked forever, waiting for each other to release the resources they need.

---

**Q48: What is an `ExecutorService`?**

**A:** A framework for managing a pool of threads, which improves performance through thread reuse.

---

**Q49: What is the difference between `Callable` and `Runnable`?**

**A:**
-   **`Runnable`**: Its `run()` method is `void` and cannot throw checked exceptions.
-   **`Callable`**: Its `call()` method returns a result (via a `Future`) and can throw exceptions.

---

**Q50: What is a `Daemon` thread?**

**A:** A low-priority thread that runs in the background. The JVM will exit when only daemon threads are left running.

---

##  Java 8+ Features

**Q51: What is a functional interface?**

**A:** An interface that contains exactly one abstract method (e.g., `Runnable`).

---

**Q52: What are Lambda expressions?**

**A:** A concise, anonymous function used to provide an implementation for a functional interface. Syntax: `(parameters) -> { body }`.

---

**Q53: What are Streams in Java 8?**

**A:** A sequence of elements from a source that supports aggregate operations (e.g., `filter`, `map`, `collect`).

---

**Q54: What is the difference between intermediate and terminal operations in Streams?**

**A:**
-   **Intermediate**: Return a new stream and are lazy (e.g., `filter()`, `map()`).
-   **Terminal**: Trigger the stream processing and produce a result (e.g., `collect()`, `forEach()`).

---

**Q55: What is the `Optional` class?**

**A:** A container object that may or may not contain a non-null value, used to avoid `NullPointerException`s.

---

**Q56: What are default methods in interfaces?**

**A:** Methods in an interface that have a default implementation, added in Java 8 to allow interfaces to evolve without breaking existing classes.

---

**Q57: What is the difference between `map()` and `flatMap()` in streams?**

**A:**
-   `map()`: Transforms each element into another object (one-to-one mapping).
-   `flatMap()`: Transforms each element into a stream and then flattens these streams into a single stream (one-to-many mapping).

---

**Q58: What are method references?**

**A:** A shorthand syntax for a lambda expression that executes a single, existing method (e.g., `String::toUpperCase`).

---

**Q59: What is the `forEach()` method?**

**A:** A terminal operation on streams (and a method on `Iterable`) that performs a given action for each element.

---

**Q60: What are Collectors?**

**A:** Used in stream terminal operations (`collect()`) to transform the elements of the stream into a result, such as a `List`, `Set`, or `Map`.

---

##  JVM, Memory & Garbage Collection

**Q61: What is garbage collection?**

**A:** The automatic process of reclaiming memory by destroying objects that are no longer referenced.

---

**Q62: What are the different memory areas in the JVM?**

**A:** Heap, Stack, Method Area, Program Counter (PC) Registers, and Native Method Stack.

---

**Q63: What is the difference between Heap and Stack memory?**

**A:**
-   **Heap**: Used for dynamic memory allocation of objects at runtime.
-   **Stack**: Used for static memory allocation and execution of threads. Contains method-specific values and references.

---

**Q64: What is JIT (Just-In-Time) compilation?**

**A:** A feature of the JVM that improves performance by compiling frequently executed bytecode into native machine code at runtime.

---

**Q65: What is the purpose of the `System.gc()` method?**

**A:** It's a suggestion to the JVM to run the Garbage Collector. There is no guarantee that the GC will actually run.

---

**Q66: What are the different types of Garbage Collectors?**

**A:** Serial GC, Parallel GC, CMS (Concurrent Mark Sweep) GC, and G1 (Garbage-First) GC.

---

##  Design Patterns, Generics & More

**Q67: What is the Singleton pattern?**

**A:** A design pattern that ensures a class has only one instance and provides a global point of access to it.

---

**Q68: What is the Factory pattern?**

**A:** A design pattern that provides an interface for creating objects in a superclass but allows subclasses to alter the type of objects that will be created.

---

**Q69: What are Generics in Java?**

**A:** A feature that provides compile-time type safety by allowing classes and methods to operate on objects of various types.

---

**Q70: What is Type Erasure?**

**A:** The process where the compiler removes all generic type information during compilation, replacing them with `Object` or their bounds.

---

**Q71: What is a wildcard (`?`) in Generics?**

**A:** Represents an unknown type. It's used to create more flexible APIs.

---

**Q72: What is Serialization?**

**A:** The process of converting a Java object's state into a byte stream. The reverse process is deserialization.

---

**Q73: What is the `transient` keyword?**

**A:** A variable modifier used to indicate that a field should not be serialized.

---

**Q74: What is reflection?**

**A:** An API that allows a program to examine or modify the runtime behavior of classes, methods, and interfaces.

---

**Q75: What are annotations?**

**A:** A form of metadata that can be added to Java code (e.g., `@Override`, `@Deprecated`).

---

**Q76: Can you override a `private` or `static` method?**

**A:** No. Private methods are not visible to subclasses, and static methods belong to the class, not the instance, so overriding doesn't apply.

---

**Q77: What is the `Object` class?**

**A:** The root of the class hierarchy. Every class in Java is a direct or indirect subclass of `Object`.

---

**Q78: What is a marker interface?**

**A:** An empty interface used to provide metadata (e.g., `Serializable`).

---

**Q79: Can a constructor be `private`?**

**A:** Yes, this is commonly used in the Singleton pattern.

---

**Q80: What are shallow copy and deep copy?**

**A:** A shallow copy shares references to nested objects, while a deep copy creates fully independent copies of all objects.

---

**Q81: What is the `instanceof` operator?**

**A:** Tests if an object is an instance of a specific type (class or interface).

---

**Q82: What is the difference between `final` and `immutable`?**

**A:** `final` is a keyword that prevents reassignment of a variable. `Immutability` is a design concept where an object's state cannot be changed after creation.

---

**Q83: What is the difference between `HashSet` and `HashMap`?**

**A:** `HashSet` stores a collection of unique elements. `HashMap` stores a collection of key-value pairs.

---

**Q84: Can an `interface` have a constructor?**

**A:** No.

---

**Q85: What are Variable Arguments (Varargs)?**

**A:** A feature allowing a method to accept zero or more arguments of the same type (`String... values`).

---

**Q86: What is thread starvation?**

**A:** When a thread is perpetually denied access to the resources it needs to run.

---

**Q87: What is the difference between `peek()` and `forEach()` in streams?**

**A:** `peek()` is an intermediate operation for debugging. `forEach()` is a terminal operation that consumes the stream.

---

**Q88: What is the `System` class?**

**A:** A final class in `java.lang` that provides access to system resources.

---

**Q89: What is the difference between `throw` and `throws` in an exception?**

**A:** `throw` is used to throw an exception, while `throws` is used to declare it in a method signature.

---

**Q90: What is a block in Java?**

**A:** A group of zero or more statements between balanced braces `{}`.

---

**Q91: What is the `this` keyword?**

**A:** A reference to the current object instance.

---

**Q92: Can you have an empty `.java` source file?**

**A:** Yes.

---

**Q93: What is a ternary operator?**

**A:** A shorthand for an `if-then-else` statement (`condition ? value_if_true : value_if_false`).

---

**Q94: Can you have a `static` class in Java?**

**A:** Yes, but only as a nested class.

---

**Q95: What is the `break` statement?**

**A:** Used to terminate a `switch` statement or a loop.

---

**Q96: What is the `continue` statement?**

**A:** Skips the current iteration of a loop and proceeds to the next one.

---

**Q97: What are the default values for primitive types?**

**A:** `0` for numeric types, `\u0000` for `char`, `false` for `boolean`.

---

**Q98: What is method chaining?**

**A:** Calling multiple methods on the same object in a single line (e.g., `string.trim().toUpperCase()`).

---

**Q99: What are Java packages used for?**

**A:** To group related classes and interfaces and to avoid naming conflicts.

---

**Q100: What is the main method signature?**

**A:** `public static void main(String[] args)`.

---