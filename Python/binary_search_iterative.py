from typing import List


def binary_search_iterative(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def _run_tests() -> None:
    arr = [1, 3, 5, 7, 9, 11]
    assert binary_search_iterative(arr, 1) == 0
    assert binary_search_iterative(arr, 11) == 5
    assert binary_search_iterative(arr, 6) == -1
    assert binary_search_iterative([], 10) == -1


if __name__ == "__main__":
    _run_tests()
    print("binary_search_iterative tests passed.")


