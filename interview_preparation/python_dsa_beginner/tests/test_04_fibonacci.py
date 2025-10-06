import pytest
from interview_preparation.python_dsa_beginner.problem_04_fibonacci import fibonacci

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(7) == 13
