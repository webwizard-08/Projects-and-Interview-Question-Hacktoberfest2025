# Question:
# You are given a set of items, each with a weight and a value, and a maximum weight capacity of a bag.
# Your goal is to determine the maximum value you can obtain by selecting any number of items (including duplicates) 
# without exceeding the weight capacity of the knapsack.
#
# Input:
# Implement the function getMaximumValue(values, weights, n, maxWeight) which takes:
# - values: List of integers representing values of the items
# - weights: List of integers representing weights of the items
# - n: Number of items
# - maxWeight: Maximum weight capacity of the bag
#
# Output:
# Return a single integer representing the maximum value that can be obtained without exceeding the weight of the bag.
#
# Constraints:
# 1 <= n <= 50
# 2 <= values[i] <= 1000
# 1 <= weights[i] <= 30
# 1 <= maxWeight <= 1000
#
# Example:
# Input:
# values = [25, 30, 15]
# weights = [15, 5, 10]
# n = 3
# maxWeight = 60
# Output: 360
# Explanation: Select twelve items of weight 5 (value 30 each) to maximize value: 12*30 = 360.

# Solution:

def getMaximumValue(values, weights, n, maxWeight):
    dp = [0] * (maxWeight + 1)
    
    for w in range(maxWeight + 1):
        for i in range(n):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[maxWeight]

# Test cases
if __name__ == "__main__":
    test_cases = [
        ([25, 30, 15], [15, 5, 10], 3, 60),   # Expected output: 360
        ([10, 40, 50, 70], [1, 3, 4, 5], 4, 8), # Expected output: 110
        ([5, 10, 15], [1, 2, 3], 3, 5),        # Expected output: 25
        ([60, 100, 120], [10, 20, 30], 3, 50)  # Expected output: 300
    ]
    
    for values, weights, n, maxWeight in test_cases:
        print(f"values={values}, weights={weights}, maxWeight={maxWeight} => maxValue={getMaximumValue(values, weights, n, maxWeight)}")
