import pytest
from interview_preparation.python_dsa_beginner.problem_06_merge_sorted_lists import merge_sorted_lists

def test_merge_sorted_lists():
    assert merge_sorted_lists([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
    assert merge_sorted_lists([], [1,2]) == [1,2]
