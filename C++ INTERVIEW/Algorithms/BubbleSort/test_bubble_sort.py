import unittest
from bubble_sort import bubble_sort

class TestBubbleSort(unittest.TestCase):

    def test_bubble_sort(self):
        self.assertEqual(bubble_sort([64, 34, 25, 12, 22, 11, 90]), [11, 12, 22, 25, 34, 64, 90])

    def test_empty_list(self):
        self.assertEqual(bubble_sort([]), [])

    def test_sorted_list(self):
        self.assertEqual(bubble_sort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_reverse_sorted_list(self):
        self.assertEqual(bubble_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_list_with_duplicates(self):
        self.assertEqual(bubble_sort([5, 4, 3, 2, 1, 5, 4, 3, 2, 1]), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

if __name__ == '__main__':
    unittest.main()
