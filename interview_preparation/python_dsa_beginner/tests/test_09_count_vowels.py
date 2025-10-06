import pytest
from interview_preparation.python_dsa_beginner.problem_09_count_vowels import count_vowels

def test_count_vowels():
    assert count_vowels("hello") == 2
    assert count_vowels("bcdfg") == 0
