from collections import deque
from typing import Deque, List


def sliding_window_max(nums: List[int], k: int) -> List[int]:
    if k <= 0:
        raise ValueError("k must be > 0")
    n = len(nums)
    if n == 0:
        return []
    if k > n:
        k = n

    dq: Deque[int] = deque()  # stores indices, decreasing values
    result: List[int] = []

    for i, val in enumerate(nums):
        # remove indices out of window
        while dq and dq[0] <= i - k:
            dq.popleft()
        # maintain decreasing deque
        while dq and nums[dq[-1]] <= val:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result


def _run_tests() -> None:
    assert sliding_window_max([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
    assert sliding_window_max([9, 11], 2) == [11]
    assert sliding_window_max([4, -2], 3) == [4]
    assert sliding_window_max([], 1) == []
    try:
        sliding_window_max([1,2,3], 0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    _run_tests()
    print("sliding_window_max tests passed.")


