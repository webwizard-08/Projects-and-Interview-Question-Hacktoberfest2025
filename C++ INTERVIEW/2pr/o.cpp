#include <iostream>
using namespace std;

int main() {
    int x = 10;          // high-level: simple variable
    int* p = &x;         // low-level: direct memory access with pointer

    *p = 20;             // modifying value through pointer
    cout << "x = " << x << endl;

    return 0;
}