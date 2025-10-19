#include <iostream>
using namespace std;

class Demo {
private:
    int a = 1;      // Not accessible outside class

protected:
    int b = 2;      // Accessible in derived class

public:
    int c = 3;      // Accessible from anywhere

    void show() {
        cout << "a = " << a << ", b = " << b << ", c = " << c << endl;
    }
};

class Derived : public Demo {
public:
    void access() {
        // cout << a << endl; Not allowed (private)
        cout << b << endl;    // Allowed (protected)
        cout << c << endl;    // Allowed (public)
    }
};

int main() {
    Derived d;        // Create object normally
    d.access();       // Call member function
    return 0;
}