import pytest
from interview_preparation.python_dsa_beginner.problem_03_factorial import factorial

def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120
