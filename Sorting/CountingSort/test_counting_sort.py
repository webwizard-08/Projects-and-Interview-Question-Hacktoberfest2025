import unittest
from counting_sort import counting_sort

class TestCountingSort(unittest.TestCase):

    def test_empty_list(self):
        self.assertEqual(counting_sort([]), [])

    def test_sorted_list(self):
        self.assertEqual(counting_sort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_reverse_sorted_list(self):
        self.assertEqual(counting_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_list_with_duplicates(self):
        self.assertEqual(counting_sort([4, 2, 2, 8, 3, 3, 1]), [1, 2, 2, 3, 3, 4, 8])

    def test_list_with_all_same_elements(self):
        self.assertEqual(counting_sort([5, 5, 5, 5, 5]), [5, 5, 5, 5, 5])

if __name__ == '__main__':
    unittest.main()
