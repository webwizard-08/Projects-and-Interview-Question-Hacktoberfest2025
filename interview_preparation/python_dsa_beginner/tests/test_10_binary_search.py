import pytest
from interview_preparation.python_dsa_beginner.problem_10_binary_search import binary_search

def test_binary_search():
    assert binary_search([1,2,3,4,5], 3) == 2
    assert binary_search([1,2,3,4,5], 6) == -1
