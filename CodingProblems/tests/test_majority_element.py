import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithms')))

from majority_element import find_majority_element

class TestMajorityElement(unittest.TestCase):

    def test_example_1(self):
        """Test with a simple example."""
        self.assertEqual(find_majority_element([3, 2, 3]), 3)

    def test_example_2(self):
        """Test with another example."""
        self.assertEqual(find_majority_element([2, 2, 1, 1, 1, 2, 2]), 2)

    def test_negative_numbers(self):
        """Test with negative numbers."""
        self.assertEqual(find_majority_element([-1, -1, -1, 2, 2]), -1)

    def test_long_array(self):
        """Test with a longer array."""
        self.assertEqual(find_majority_element([1, 1, 1, 1, 1, 2, 3, 4, 5]), 1)

if __name__ == '__main__':
    unittest.main()
