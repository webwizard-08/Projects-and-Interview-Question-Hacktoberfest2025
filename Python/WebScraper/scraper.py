
import requests
from bs4 import BeautifulSoup

def get_title(url):
    """
    Fetches the title of a given URL.

    Args:
        url: The URL to fetch the title from.

    Returns:
        The title of the URL, or None if the title cannot be found.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.title.string
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

if __name__ == "__main__":
    test_url = "http://books.toscrape.com/"
    title = get_title(test_url)
    if title:
        print(f"The title of the website is: {title}")
    else:
        print("Could not retrieve the title.")
