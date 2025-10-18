/*
 * C++ Interview Questions - Code Examples
 * 
 * This file contains practical code examples for C++ interview questions
 * covering Object-Oriented Programming, STL, and Memory Management.
 * 
 * Compile with: g++ -std=c++17 -Wall -Wextra Code_Examples.cpp -o examples
 * Run with: ./examples
 */

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <list>
#include <string>
#include <fstream>
#include <exception>

// ============================================================================
// OBJECT-ORIENTED PROGRAMMING EXAMPLES
// ============================================================================

// Example 1: Complete OOP Implementation - Bank Account System
class BankAccount {
private:
    std::string accountNumber;
    double balance;
    static int totalAccounts; // Static member

public:
    // Constructor
    BankAccount(const std::string& accNum, double initialBalance = 0.0)
        : accountNumber(accNum), balance(initialBalance) {
        totalAccounts++;
        std::cout << "Account " << accountNumber << " created with balance: $" 
                  << balance << std::endl;
    }

    // Destructor
    ~BankAccount() {
        totalAccounts--;
        std::cout << "Account " << accountNumber << " closed" << std::endl;
    }

    // Copy constructor
    BankAccount(const BankAccount& other)
        : accountNumber(other.accountNumber + "_copy"), balance(other.balance) {
        totalAccounts++;
        std::cout << "Account copied: " << accountNumber << std::endl;
    }

    // Assignment operator
    BankAccount& operator=(const BankAccount& other) {
        if (this != &other) {
            accountNumber = other.accountNumber + "_assigned";
            balance = other.balance;
        }
        return *this;
    }

    // Getters
    const std::string& getAccountNumber() const { return accountNumber; }
    double getBalance() const { return balance; }
    static int getTotalAccounts() { return totalAccounts; }

    // Methods
    bool deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            std::cout << "Deposited $" << amount << ". New balance: $" << balance << std::endl;
            return true;
        }
        return false;
    }

    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            std::cout << "Withdrew $" << amount << ". New balance: $" << balance << std::endl;
            return true;
        }
        std::cout << "Insufficient funds or invalid amount" << std::endl;
        return false;
    }

    // Virtual function for polymorphism
    virtual void displayInfo() const {
        std::cout << "Account: " << accountNumber << ", Balance: $" << balance << std::endl;
    }
};

// Static member definition
int BankAccount::totalAccounts = 0;

// Derived class - Savings Account
class SavingsAccount : public BankAccount {
private:
    double interestRate;

public:
    SavingsAccount(const std::string& accNum, double initialBalance, double rate)
        : BankAccount(accNum, initialBalance), interestRate(rate) {}

    void addInterest() {
        double interest = getBalance() * interestRate / 100;
        deposit(interest);
        std::cout << "Interest added: $" << interest << std::endl;
    }

    void displayInfo() const override {
        BankAccount::displayInfo();
        std::cout << "Interest Rate: " << interestRate << "%" << std::endl;
    }
};

// ============================================================================
// STL EXAMPLES
// ============================================================================

class STLExamples {
public:
    // Vector operations
    static void demonstrateVector() {
        std::cout << "\n=== VECTOR DEMONSTRATION ===" << std::endl;
        
        std::vector<int> numbers = {5, 2, 8, 1, 9, 3};
        
        // Add elements
        numbers.push_back(7);
        numbers.insert(numbers.begin() + 2, 4);
        
        // Sort
        std::sort(numbers.begin(), numbers.end());
        
        // Display
        std::cout << "Sorted vector: ";
        for (const auto& num : numbers) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
        
        // Find element
        auto it = std::find(numbers.begin(), numbers.end(), 5);
        if (it != numbers.end()) {
            std::cout << "Found 5 at position: " << std::distance(numbers.begin(), it) << std::endl;
        }
    }

    // Map operations
    static void demonstrateMap() {
        std::cout << "\n=== MAP DEMONSTRATION ===" << std::endl;
        
        std::map<std::string, int> wordCount;
        std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};
        
        // Count word frequencies
        for (const auto& word : words) {
            wordCount[word]++;
        }
        
        // Display results
        std::cout << "Word frequencies:" << std::endl;
        for (const auto& pair : wordCount) {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
    }

    // Algorithm examples
    static void demonstrateAlgorithms() {
        std::cout << "\n=== ALGORITHMS DEMONSTRATION ===" << std::endl;
        
        std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        
        // Count even numbers
        int evenCount = std::count_if(numbers.begin(), numbers.end(),
                                     [](int n) { return n % 2 == 0; });
        std::cout << "Even numbers count: " << evenCount << std::endl;
        
        // Transform (square all numbers)
        std::vector<int> squares;
        std::transform(numbers.begin(), numbers.end(), std::back_inserter(squares),
                      [](int n) { return n * n; });
        
        std::cout << "Squares: ";
        for (int num : squares) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
        
        // Accumulate (sum)
        int sum = std::accumulate(numbers.begin(), numbers.end(), 0);
        std::cout << "Sum of numbers: " << sum << std::endl;
    }
};

// ============================================================================
// MEMORY MANAGEMENT EXAMPLES
// ============================================================================

// RAII Example - File Manager
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

    // Disable copy
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
};

// Smart Pointer Examples
class Resource {
private:
    int id;
    static int nextId;

public:
    Resource() : id(++nextId) {
        std::cout << "Resource " << id << " created" << std::endl;
    }

    ~Resource() {
        std::cout << "Resource " << id << " destroyed" << std::endl;
    }

    void doWork() {
        std::cout << "Resource " << id << " is working" << std::endl;
    }

    int getId() const { return id; }
};

int Resource::nextId = 0;

// Node for demonstrating shared_ptr
class Node {
public:
    int data;
    std::vector<std::shared_ptr<Node>> children;
    std::weak_ptr<Node> parent; // Use weak_ptr to avoid circular reference

    Node(int d) : data(d) {
        std::cout << "Node " << data << " created" << std::endl;
    }

    ~Node() {
        std::cout << "Node " << data << " destroyed" << std::endl;
    }

    void addChild(std::shared_ptr<Node> child) {
        children.push_back(child);
        child->parent = shared_from_this();
    }
};

// ============================================================================
// MAIN FUNCTION - DEMONSTRATION
// ============================================================================

int main() {
    std::cout << "=== C++ INTERVIEW QUESTIONS - CODE EXAMPLES ===" << std::endl;

    // OOP Examples
    std::cout << "\n=== OBJECT-ORIENTED PROGRAMMING ===" << std::endl;
    
    // Bank Account System
    BankAccount account1("ACC001", 1000.0);
    account1.deposit(500.0);
    account1.withdraw(200.0);
    account1.displayInfo();
    
    // Savings Account
    SavingsAccount savings("SAV001", 2000.0, 2.5);
    savings.addInterest();
    savings.displayInfo();
    
    // Copy constructor and assignment
    BankAccount account2 = account1; // Copy constructor
    BankAccount account3("ACC003", 500.0);
    account3 = account1; // Assignment operator
    
    std::cout << "Total accounts created: " << BankAccount::getTotalAccounts() << std::endl;

    // STL Examples
    STLExamples::demonstrateVector();
    STLExamples::demonstrateMap();
    STLExamples::demonstrateAlgorithms();

    // Memory Management Examples
    std::cout << "\n=== MEMORY MANAGEMENT ===" << std::endl;
    
    // RAII Example
    try {
        FileManager fm("test_output.txt");
        fm.write("This is a test file created using RAII");
        fm.write("The file will be automatically closed when the object is destroyed");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Smart Pointers
    std::cout << "\n--- Smart Pointers Demo ---" << std::endl;
    
    // Unique pointer
    {
        std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>();
        ptr1->doWork();
        
        // Transfer ownership
        std::unique_ptr<Resource> ptr2 = std::move(ptr1);
        if (!ptr1) {
            std::cout << "ptr1 is now null after move" << std::endl;
        }
        ptr2->doWork();
    } // Resource automatically destroyed

    // Shared pointer
    std::cout << "\n--- Shared Pointer Demo ---" << std::endl;
    {
        auto node1 = std::make_shared<Node>(1);
        auto node2 = std::make_shared<Node>(2);
        auto node3 = std::make_shared<Node>(3);
        
        node1->addChild(node2);
        node1->addChild(node3);
        
        std::cout << "Node1 use count: " << node1.use_count() << std::endl;
        std::cout << "Node2 use count: " << node2.use_count() << std::endl;
    } // All nodes automatically destroyed

    std::cout << "\n=== DEMONSTRATION COMPLETE ===" << std::endl;
    return 0;
}

/*
 * COMPILATION AND RUNNING INSTRUCTIONS:
 * 
 * 1. Compile: g++ -std=c++17 -Wall -Wextra Code_Examples.cpp -o examples
 * 2. Run: ./examples
 * 
 * EXPECTED OUTPUT:
 * - Bank account operations
 * - STL container and algorithm demonstrations
 * - RAII file management
 * - Smart pointer lifecycle management
 * - Automatic resource cleanup
 * 
 * KEY LEARNING POINTS:
 * 1. OOP: Encapsulation, inheritance, polymorphism, abstraction
 * 2. STL: Containers, algorithms, iterators, lambdas
 * 3. Memory Management: RAII, smart pointers, exception safety
 * 4. Modern C++: auto, range-based loops, move semantics
 */
