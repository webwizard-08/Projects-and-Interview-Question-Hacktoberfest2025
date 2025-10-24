import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
from life_game import main

class TestLifeGame(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_axes_are_visible(self, mock_show):
        main()
        fig = plt.gcf()
        ax = fig.gca()
        self.assertEqual(ax.get_title(), "Conway's Game of Life")
        self.assertTrue(ax.axison)

if __name__ == '__main__':
    unittest.main()
