#include <iostream>
using namespace std;

int main() {
    try {
        throw runtime_error("Error occurred");
    }
    catch (const runtime_error& e) {
        cout << "Caught exception: " << e.what() << endl;
    }
}