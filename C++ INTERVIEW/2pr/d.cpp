#include <iostream>
using namespace std;

// Base with virtual and pure virtual
class Base {
public:
    virtual void greet() {  // Virtual function
        cout << "Hello from Base" << endl;
    }

    virtual void pureGreet() = 0;  // Pure virtual function
};

// Derived implements pure virtual
class Derived : public Base {
public:
    void greet() override {
        cout << "Hello from Derived (Override)" << endl;
    }

    void pureGreet() override {
        cout << "Hello from Derived (Pure Virtual Implemented)" << endl;
    }
};

int main() {

    Derived d;
    Base* ptr = &d;

    ptr->greet();      // Calls overridden version
    ptr->pureGreet();  // Calls derived's implementation

    return 0;
}