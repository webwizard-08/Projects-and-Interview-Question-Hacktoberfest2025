#include <vector>
#include <iostream>
using namespace std;

int main() {
    vector<int> v = {10, 20, 30};
    v.push_back(40); // add at end
    v.pop_back();    // remove last
    cout << v[1];    // random access
}