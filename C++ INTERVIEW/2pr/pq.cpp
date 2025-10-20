#include <iostream>
using namespace std;

class Animal {
public:
    // Virtual function allows overriding
    virtual void sound() {
        cout << "Animal makes a sound" << endl;
    }
};

class Dog : public Animal {
public:
    // Overrides the base class function
    void sound() override {
        cout << "Dog barks" << endl;
    }
};

int main() {
    Animal* a;       // Base class pointer
    Dog d;           // Derived class object
    a = &d;          // Base pointer points to derived object

    a->sound();      // Runtime polymorphism: Dog's version called
    return 0;
}