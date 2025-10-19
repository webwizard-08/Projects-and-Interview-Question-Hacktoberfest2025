#include <iostream>
#include <exception>
using namespace std;

class MyException : public exception {
public:
    const char* what() const noexcept override {
        return "Custom Exception Occurred";
    }
};

int main() {
    try {
        throw MyException();
    } catch (const MyException& e) {
        cout << e.what();
    }
}