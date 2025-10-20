//Example for struct and class

#include <iostream>
using namespace std;

struct MyStruct {
    int x;            // public by default
    void show() {
        cout << "Struct x = " << x << endl;
    }
};

class MyClass {
    int y;            // private by default
public:
    MyClass(int val) : y(val) {}   // constructor
    void show() {
        cout << "Class y = " << y << endl;
    }
};

int main() {
    MyStruct s; 
    s.x = 10;      // allowed (public by default)
    s.show();

    MyClass c(20);
    // c.y = 20;   // not allowed, y is private
    c.show();      // access via public function
}