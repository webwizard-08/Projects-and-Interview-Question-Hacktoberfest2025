from typing import Dict, List, Tuple


def two_sum(nums: List[int], target: int) -> Tuple[int, int]:
    index_by_value: Dict[int, int] = {}
    for i, num in enumerate(nums):
        need = target - num
        if need in index_by_value:
            return index_by_value[need], i
        index_by_value[num] = i
    raise ValueError("No two sum solution exists for the given input")


def _run_tests() -> None:
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum([3, 2, 4], 6) == (1, 2)
    assert two_sum([3, 3], 6) == (0, 1)
    try:
        two_sum([1, 2, 3], 100)
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    _run_tests()
    print("two_sum tests passed.")


