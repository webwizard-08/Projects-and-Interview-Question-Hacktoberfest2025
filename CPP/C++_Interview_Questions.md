# 🚀 C++ Interview Questions Guide

Welcome to the **C++ Interview Questions** repository!  
This guide contains a comprehensive collection of **interview-focused questions and answers** covering **Object-Oriented Programming**, **STL (Standard Template Library)**, and **Memory Management** topics — all frequently asked in **FAANG**, **tech companies**, and **C++ developer** interviews.

---

## 🏗️ Object-Oriented Programming (OOP)

### Q1. Explain the four pillars of Object-Oriented Programming with code examples.

**A:** The four pillars of OOP are:

#### 1. **Encapsulation** - Data Hiding
```cpp
#include <iostream>
#include <string>

class BankAccount {
private:
    double balance;
    std::string accountNumber;
    
public:
    BankAccount(const std::string& accNum, double initialBalance) 
        : accountNumber(accNum), balance(initialBalance) {}
    
    // Getter methods
    double getBalance() const { return balance; }
    std::string getAccountNumber() const { return accountNumber; }
    
    // Setter methods with validation
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }
    
    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
};

int main() {
    BankAccount account("12345", 1000.0);
    account.deposit(500.0);
    std::cout << "Balance: $" << account.getBalance() << std::endl;
    return 0;
}
```

#### 2. **Inheritance** - Code Reusability
```cpp
#include <iostream>
#include <string>

// Base class
class Vehicle {
protected:
    std::string brand;
    int year;
    
public:
    Vehicle(const std::string& b, int y) : brand(b), year(y) {}
    
    virtual void start() {
        std::cout << "Vehicle is starting..." << std::endl;
    }
    
    virtual void displayInfo() {
        std::cout << "Brand: " << brand << ", Year: " << year << std::endl;
    }
    
    virtual ~Vehicle() = default; // Virtual destructor
};

// Derived class
class Car : public Vehicle {
private:
    int doors;
    
public:
    Car(const std::string& b, int y, int d) : Vehicle(b, y), doors(d) {}
    
    void start() override {
        std::cout << "Car engine is starting..." << std::endl;
    }
    
    void displayInfo() override {
        Vehicle::displayInfo();
        std::cout << "Doors: " << doors << std::endl;
    }
};

int main() {
    Car myCar("Toyota", 2023, 4);
    myCar.start();
    myCar.displayInfo();
    return 0;
}
```

#### 3. **Polymorphism** - Runtime Binding
```cpp
#include <iostream>
#include <vector>
#include <memory>

class Shape {
public:
    virtual double area() const = 0; // Pure virtual function
    virtual void draw() const {
        std::cout << "Drawing a shape" << std::endl;
    }
    virtual ~Shape() = default;
};

class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(double r) : radius(r) {}
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
    
    void draw() const override {
        std::cout << "Drawing a circle with radius " << radius << std::endl;
    }
};

class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    double area() const override {
        return width * height;
    }
    
    void draw() const override {
        std::cout << "Drawing a rectangle " << width << "x" << height << std::endl;
    }
};

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 6.0));
    
    for (const auto& shape : shapes) {
        shape->draw();
        std::cout << "Area: " << shape->area() << std::endl;
    }
    return 0;
}
```

#### 4. **Abstraction** - Interface Definition
```cpp
#include <iostream>
#include <memory>

// Abstract base class (Interface)
class Database {
public:
    virtual bool connect() = 0;
    virtual bool disconnect() = 0;
    virtual std::string query(const std::string& sql) = 0;
    virtual ~Database() = default;
};

// Concrete implementation
class MySQLDatabase : public Database {
public:
    bool connect() override {
        std::cout << "Connecting to MySQL database..." << std::endl;
        return true;
    }
    
    bool disconnect() override {
        std::cout << "Disconnecting from MySQL database..." << std::endl;
        return true;
    }
    
    std::string query(const std::string& sql) override {
        std::cout << "Executing MySQL query: " << sql << std::endl;
        return "Query result from MySQL";
    }
};

class PostgreSQLDatabase : public Database {
public:
    bool connect() override {
        std::cout << "Connecting to PostgreSQL database..." << std::endl;
        return true;
    }
    
    bool disconnect() override {
        std::cout << "Disconnecting from PostgreSQL database..." << std::endl;
        return true;
    }
    
    std::string query(const std::string& sql) override {
        std::cout << "Executing PostgreSQL query: " << sql << std::endl;
        return "Query result from PostgreSQL";
    }
};

int main() {
    std::unique_ptr<Database> db = std::make_unique<MySQLDatabase>();
    db->connect();
    std::string result = db->query("SELECT * FROM users");
    db->disconnect();
    return 0;
}
```

---

## 📚 STL (Standard Template Library)

### Q2. Explain the key STL containers and their use cases with examples.

**A:** STL provides several container types, each optimized for different use cases:

#### 1. **Sequence Containers**

**Vector** - Dynamic Array
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {5, 2, 8, 1, 9};
    
    // Adding elements
    numbers.push_back(3);
    numbers.insert(numbers.begin() + 2, 7);
    
    // Accessing elements
    std::cout << "First element: " << numbers[0] << std::endl;
    std::cout << "Size: " << numbers.size() << std::endl;
    
    // Sorting
    std::sort(numbers.begin(), numbers.end());
    
    // Iterating
    std::cout << "Sorted numbers: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

**List** - Doubly Linked List
```cpp
#include <iostream>
#include <list>

int main() {
    std::list<int> myList = {1, 2, 3, 4, 5};
    
    // Insert at beginning and end
    myList.push_front(0);
    myList.push_back(6);
    
    // Insert in middle
    auto it = myList.begin();
    std::advance(it, 3);
    myList.insert(it, 99);
    
    // Remove elements
    myList.remove(3); // Remove all occurrences of 3
    
    // Display
    std::cout << "List contents: ";
    for (const auto& val : myList) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### 2. **Associative Containers**

**Map** - Key-Value Pairs (Ordered)
```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<std::string, int> ageMap;
    
    // Inserting elements
    ageMap["Alice"] = 25;
    ageMap["Bob"] = 30;
    ageMap["Charlie"] = 35;
    
    // Accessing elements
    std::cout << "Alice's age: " << ageMap["Alice"] << std::endl;
    
    // Checking if key exists
    if (ageMap.find("David") != ageMap.end()) {
        std::cout << "David found!" << std::endl;
    } else {
        std::cout << "David not found!" << std::endl;
    }
    
    // Iterating through map
    std::cout << "All ages:" << std::endl;
    for (const auto& pair : ageMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    return 0;
}
```

**Unordered Map** - Hash Table
```cpp
#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::unordered_map<std::string, int> wordCount;
    
    // Counting word frequencies
    std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};
    
    for (const auto& word : words) {
        wordCount[word]++;
    }
    
    // Display results
    std::cout << "Word frequencies:" << std::endl;
    for (const auto& pair : wordCount) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    return 0;
}
```

#### 3. **STL Algorithms**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    // Sorting
    std::sort(numbers.begin(), numbers.end());
    
    // Finding elements
    auto it = std::find(numbers.begin(), numbers.end(), 7);
    if (it != numbers.end()) {
        std::cout << "Found 7 at position: " << std::distance(numbers.begin(), it) << std::endl;
    }
    
    // Counting
    int count = std::count_if(numbers.begin(), numbers.end(), 
                             [](int n) { return n > 5; });
    std::cout << "Numbers greater than 5: " << count << std::endl;
    
    // Accumulate (sum)
    int sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    std::cout << "Sum of all numbers: " << sum << std::endl;
    
    // Transform
    std::vector<int> doubled;
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(doubled),
                   [](int n) { return n * 2; });
    
    std::cout << "Doubled numbers: ";
    for (int num : doubled) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

---

## 🧠 Memory Management

### Q3. Explain smart pointers and RAII with practical examples.

**A:** Smart pointers and RAII are crucial for safe memory management in C++:

#### 1. **RAII (Resource Acquisition Is Initialization)**
```cpp
#include <iostream>
#include <memory>
#include <fstream>

class FileManager {
private:
    std::unique_ptr<std::fstream> file;
    std::string filename;
    
public:
    FileManager(const std::string& name) : filename(name) {
        file = std::make_unique<std::fstream>(filename, std::ios::out);
        if (!file->is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::cout << "File opened: " << filename << std::endl;
    }
    
    ~FileManager() {
        if (file && file->is_open()) {
            file->close();
            std::cout << "File closed: " << filename << std::endl;
        }
    }
    
    void write(const std::string& data) {
        if (file && file->is_open()) {
            *file << data << std::endl;
        }
    }
    
    // Disable copy constructor and assignment
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
};

int main() {
    try {
        FileManager fm("test.txt");
        fm.write("Hello, RAII!");
        fm.write("This file will be automatically closed.");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    // File is automatically closed when fm goes out of scope
    return 0;
}
```

#### 2. **Smart Pointers**

**Unique Pointer** - Exclusive Ownership
```cpp
#include <iostream>
#include <memory>

class Resource {
private:
    int id;
    
public:
    Resource(int i) : id(i) {
        std::cout << "Resource " << id << " created" << std::endl;
    }
    
    ~Resource() {
        std::cout << "Resource " << id << " destroyed" << std::endl;
    }
    
    void doSomething() {
        std::cout << "Resource " << id << " is working" << std::endl;
    }
};

int main() {
    {
        std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>(1);
        ptr1->doSomething();
        
        // Transfer ownership
        std::unique_ptr<Resource> ptr2 = std::move(ptr1);
        
        if (!ptr1) {
            std::cout << "ptr1 is now null" << std::endl;
        }
        
        ptr2->doSomething();
    } // Resource is automatically destroyed here
    
    return 0;
}
```

**Shared Pointer** - Shared Ownership
```cpp
#include <iostream>
#include <memory>
#include <vector>

class Node {
public:
    int data;
    std::vector<std::shared_ptr<Node>> children;
    
    Node(int d) : data(d) {
        std::cout << "Node " << data << " created" << std::endl;
    }
    
    ~Node() {
        std::cout << "Node " << data << " destroyed" << std::endl;
    }
    
    void addChild(std::shared_ptr<Node> child) {
        children.push_back(child);
    }
};

int main() {
    // Create nodes
    auto node1 = std::make_shared<Node>(1);
    auto node2 = std::make_shared<Node>(2);
    auto node3 = std::make_shared<Node>(3);
    
    // Build tree structure
    node1->addChild(node2);
    node1->addChild(node3);
    
    std::cout << "Node1 use count: " << node1.use_count() << std::endl;
    std::cout << "Node2 use count: " << node2.use_count() << std::endl;
    
    // Reset one reference
    node2.reset();
    std::cout << "After resetting node2, Node2 use count: " << node2.use_count() << std::endl;
    
    return 0;
    // All nodes are automatically destroyed when their use count reaches 0
}
```

**Weak Pointer** - Non-owning Reference
```cpp
#include <iostream>
#include <memory>

class Parent;
class Child;

class Parent {
public:
    std::shared_ptr<Child> child;
    ~Parent() { std::cout << "Parent destroyed" << std::endl; }
};

class Child {
public:
    std::weak_ptr<Parent> parent; // Use weak_ptr to break circular reference
    ~Child() { std::cout << "Child destroyed" << std::endl; }
    
    void checkParent() {
        if (auto parentPtr = parent.lock()) {
            std::cout << "Parent is still alive" << std::endl;
        } else {
            std::cout << "Parent has been destroyed" << std::endl;
        }
    }
};

int main() {
    {
        auto parent = std::make_shared<Parent>();
        auto child = std::make_shared<Child>();
        
        parent->child = child;
        child->parent = parent;
        
        child->checkParent();
    } // Both parent and child are destroyed properly
    
    return 0;
}
```

#### 3. **Memory Leak Prevention**
```cpp
#include <iostream>
#include <memory>
#include <vector>

// BAD: Manual memory management (prone to leaks)
class BadExample {
private:
    int* data;
    
public:
    BadExample(int size) : data(new int[size]) {
        std::cout << "Allocated " << size << " integers" << std::endl;
    }
    
    ~BadExample() {
        delete[] data; // What if exception is thrown before this?
        std::cout << "Freed memory" << std::endl;
    }
};

// GOOD: RAII with smart pointers
class GoodExample {
private:
    std::unique_ptr<int[]> data;
    size_t size;
    
public:
    GoodExample(size_t s) : data(std::make_unique<int[]>(s)), size(s) {
        std::cout << "Allocated " << size << " integers safely" << std::endl;
    }
    
    // No need for explicit destructor - unique_ptr handles it
    // Exception safe - memory is automatically freed
};

int main() {
    try {
        GoodExample good(1000);
        // Even if an exception is thrown here, memory is automatically freed
        throw std::runtime_error("Something went wrong!");
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        // Memory is still properly cleaned up
    }
    
    return 0;
}
```

---

## 🎯 Interview Tips

### Key Points to Remember:

1. **OOP Concepts**: Always explain with concrete examples, not just definitions
2. **STL Proficiency**: Know when to use which container and algorithm
3. **Memory Safety**: Emphasize RAII and smart pointers over raw pointers
4. **Modern C++**: Use C++11/14/17/20 features when appropriate
5. **Performance**: Understand time/space complexity of STL operations

### Common Follow-up Questions:

- "What's the difference between `std::vector` and `std::array`?"
- "When would you use `std::shared_ptr` vs `std::unique_ptr`?"
- "How does virtual function dispatch work?"
- "What are the benefits of using STL algorithms over manual loops?"

---

**Happy Learning! 🚀**
