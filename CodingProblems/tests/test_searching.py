import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithms')))

from searching import two_sum

class TestTwoSum(unittest.TestCase):

    def test_example_1(self):
        """Test with the first example case."""
        self.assertEqual(sorted(two_sum([2, 7, 11, 15], 9)), [0, 1])

    def test_example_2(self):
        """Test with the second example case."""
        self.assertEqual(sorted(two_sum([3, 2, 4], 6)), [1, 2])

    def test_no_solution(self):
        """Test with a case where no solution exists."""
        self.assertIsNone(two_sum([1, 2, 3], 7))

    def test_negative_numbers(self):
        """Test with negative numbers."""
        self.assertEqual(sorted(two_sum([-1, -3, 5, 9], 4)), [0, 2])

    def test_zero(self):
        """Test with zero in the list."""
        self.assertEqual(sorted(two_sum([0, 4, 3, 0], 0)), [0, 3])

if __name__ == '__main__':
    unittest.main()
