import pytest
from interview_preparation.python_dsa_beginner.problem_01_reverse_string import reverse_string

def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("a") == "a"
