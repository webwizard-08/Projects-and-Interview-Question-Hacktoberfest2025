import pytest
from interview_preparation.python_dsa_beginner.problem_02_is_prime import is_prime

def test_is_prime():
    assert is_prime(0) == False
    assert is_prime(2) == True
    assert is_prime(15) == False
