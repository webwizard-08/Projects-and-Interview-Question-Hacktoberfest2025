import pytest
from interview_preparation.python_dsa_beginner.problem_08_valid_parentheses import is_valid_parentheses

def test_valid_parentheses():
    assert is_valid_parentheses("()[]{}") == True
    assert is_valid_parentheses("(]") == False
