import pytest
from interview_preparation.python_dsa_beginner.problem_07_remove_duplicates import remove_duplicates

def test_remove_duplicates():
    assert remove_duplicates([1,2,2,3]) == [1,2,3]
    assert remove_duplicates([1,1,1]) == [1]
