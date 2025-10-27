import unittest
from branch_and_bound import knapsack_solution, Item

class TestBranchAndBound(unittest.TestCase):

    def test_knapsack_solution(self):
        W = 10
        arr = [Item(2, 40), Item(3.14, 50), Item(
            1.98, 100), Item(5, 95), Item(3, 30)]
        n = len(arr)
        self.assertEqual(knapsack_solution(W, arr, n), 235)

    def test_knapsack_solution_simple(self):
        W = 50
        arr = [Item(10, 60), Item(20, 100), Item(30, 120)]
        n = len(arr)
        self.assertEqual(knapsack_solution(W, arr, n), 220)

if __name__ == '__main__':
    unittest.main()
