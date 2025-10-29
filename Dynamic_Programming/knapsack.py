from typing import List, Tuple
import time

class KnapsackSolver:
    """
    A class that implements different solutions to the 0/1 Knapsack problem.
    The problem: Given weights and values of n items, put these items in a knapsack of 
    capacity W to get the maximum total value in the knapsack.
    """
    
    @staticmethod
    def knapsack_dp(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int]]:
        """
        Solves the 0/1 Knapsack problem using dynamic programming.
        
        Args:
            values (List[int]): List of values for each item
            weights (List[int]): List of weights for each item
            capacity (int): Maximum weight capacity of knapsack
            
        Returns:
            Tuple[int, List[int]]: Maximum value and list of selected items
            
        Time Complexity: O(n * W) where n is number of items and W is capacity
        Space Complexity: O(n * W)
        """
        n = len(values)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Fill dp table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(
                        values[i-1] + dp[i-1][w-weights[i-1]],
                        dp[i-1][w]
                    )
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Backtrack to find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= weights[i-1]
                
        return dp[n][capacity], selected[::-1]
    
    @staticmethod
    def knapsack_recursive(values: List[int], weights: List[int], capacity: int) -> int:
        """
        Recursive solution to the Knapsack problem (for comparison).
        Warning: This is not efficient for large inputs.
        
        Args:
            values (List[int]): List of values for each item
            weights (List[int]): List of weights for each item
            capacity (int): Maximum weight capacity of knapsack
            
        Returns:
            int: Maximum value possible
            
        Time Complexity: O(2^n)
        Space Complexity: O(n) due to recursion stack
        """
        def recurse(index: int, remaining_capacity: int) -> int:
            if index < 0 or remaining_capacity <= 0:
                return 0
                
            # Can't include this item
            if weights[index] > remaining_capacity:
                return recurse(index - 1, remaining_capacity)
                
            # Try including and excluding current item
            return max(
                values[index] + recurse(index - 1, remaining_capacity - weights[index]),
                recurse(index - 1, remaining_capacity)
            )
            
        return recurse(len(values) - 1, capacity)

def benchmark_knapsack(solver: KnapsackSolver, values: List[int], weights: List[int], capacity: int) -> float:
    """Benchmark a knapsack solution"""
    start = time.time()
    result = solver.knapsack_dp(values, weights, capacity)
    end = time.time()
    return end - start

# Test cases
if __name__ == "__main__":
    solver = KnapsackSolver()
    
    # Test case 1: Basic example
    values1 = [60, 100, 120]
    weights1 = [10, 20, 30]
    capacity1 = 50
    
    max_value1, selected1 = solver.knapsack_dp(values1, weights1, capacity1)
    assert max_value1 == 220
    assert set(selected1) == {1, 2}  # Items with values 100 and 120
    
    # Test case 2: No items fit
    values2 = [10, 20, 30]
    weights2 = [100, 200, 300]
    capacity2 = 50
    
    max_value2, selected2 = solver.knapsack_dp(values2, weights2, capacity2)
    assert max_value2 == 0
    assert len(selected2) == 0
    
    # Test case 3: All items fit
    values3 = [10, 20, 30]
    weights3 = [1, 2, 3]
    capacity3 = 10
    
    max_value3, selected3 = solver.knapsack_dp(values3, weights3, capacity3)
    assert max_value3 == 60
    assert len(selected3) == 3
    
    # Test case 4: Compare with recursive solution
    values4 = [50, 70, 100, 30]
    weights4 = [5, 7, 10, 3]
    capacity4 = 15
    
    max_value4, _ = solver.knapsack_dp(values4, weights4, capacity4)
    recursive_result = solver.knapsack_recursive(values4, weights4, capacity4)
    assert max_value4 == recursive_result
    
    print("All knapsack problem tests passed!")
    
    # Bonus: Simple performance comparison
    large_values = list(range(1, 101))
    large_weights = list(range(1, 101))
    large_capacity = 500
    
    time_taken = benchmark_knapsack(solver, large_values, large_weights, large_capacity)
    print(f"\nTime taken for large input (100 items): {time_taken:.4f} seconds")