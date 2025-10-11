import pytest
from interview_preparation.python_dsa_beginner.problem_05_two_sum import two_sum

def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([1, 2, 3], 7) == None
