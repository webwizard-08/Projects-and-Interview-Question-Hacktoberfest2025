
import unittest
from unittest.mock import patch, Mock
import requests
from scraper import get_title

class TestScraper(unittest.TestCase):

    @patch('scraper.requests.get')
    def test_get_title_success(self, mock_get):
        # Create a mock response object
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test Title</title></head><body></body></html>"
        mock_get.return_value = mock_response

        # Call the function with a test URL
        title = get_title("http://test.com")
        self.assertEqual(title, "Test Title")

    @patch('scraper.requests.get')
    def test_get_title_failure(self, mock_get):
        # Configure the mock to raise an exception
        mock_get.side_effect = requests.exceptions.RequestException("Test Error")

        # Call the function with a test URL
        title = get_title("http://test.com")
        self.assertIsNone(title)

if __name__ == '__main__':
    unittest.main()
