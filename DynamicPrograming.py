#include <iostream>
#include <vector>
using namespace std;

// Example: Fibonacci sequence with Dynamic Programming
int fibonacci(int n) {
    vector<int> dp(n + 1, 0);
    dp[0] = 0; // base case
    dp[1] = 1; // base case
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i-1] + dp[i-2]; // DP relation
    }
    return dp[n];
}

int main() {
    int n;
    cout << "Enter n for Fibonacci: ";
    cin >> n;
    cout << "Fibonacci number at position " << n << " is: " << fibonacci(n) << endl;
    return 0;
}
