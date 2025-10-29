"""
Greedy algorithm coin change examples.

This file provides:
- a greedy coin change implementation (works for canonical coin systems like US coins)
- a dynamic programming solver to compute the optimal (minimum coins) solution for verification
- example usage showing when greedy is optimal and when it's not

Author: added for Hacktoberfest repository
"""

from typing import List, Tuple


def greedy_coin_change(coins: List[int], amount: int) -> List[int]:
    """Return a list of counts of each coin (same order as `coins`) using a greedy approach.

    coins should be provided in descending order for the usual greedy loop.

    Returns a list of non-negative integers representing how many of each coin was used.
    If the amount cannot be formed with the given coins, raises ValueError.

    Complexity: O(len(coins)).
    """
    if amount < 0:
        raise ValueError("amount must be non-negative")
    counts = [0] * len(coins)
    remaining = amount
    for i, c in enumerate(coins):
        if c <= 0:
            raise ValueError("coin values must be positive")
        take = remaining // c
        if take:
            counts[i] = take
            remaining -= take * c
    if remaining != 0:
        raise ValueError(f"cannot form amount {amount} with given coins")
    return counts


def dp_coin_change_min_coins(coins: List[int], amount: int) -> Tuple[int, List[int]]:
    """Compute minimum coin count and one possible coin usage vector using DP.

    Returns (min_coins, counts_list). If impossible, returns (inf, []).
    Complexity: O(amount * len(coins)).
    """
    INF = 10 ** 9
    n = len(coins)
    # dp[x] = minimum coins to make x
    dp = [INF] * (amount + 1)
    # choice[x] = index of coin chosen last to achieve dp[x]
    choice = [-1] * (amount + 1)
    dp[0] = 0
    for i, coin in enumerate(coins):
        for x in range(coin, amount + 1):
            if dp[x - coin] + 1 < dp[x]:
                dp[x] = dp[x - coin] + 1
                choice[x] = i

    if dp[amount] >= INF:
        return INF, []

    # reconstruct counts
    counts = [0] * n
    x = amount
    while x > 0:
        idx = choice[x]
        if idx == -1:
            # shouldn't happen
            break
        counts[idx] += 1
        x -= coins[idx]

    return dp[amount], counts


def counts_to_str(coins: List[int], counts: List[int]) -> str:
    pairs = [f"{c}x{v}" for v, c in zip(coins, counts) if c]
    return ", ".join(pairs) if pairs else "(no coins)"


if __name__ == "__main__":
    # Examples
    print("Greedy coin change examples")

    # Canonical coin system (US-style): greedy is optimal
    coins_us = [25, 10, 5, 1]
    amount = 63
    greedy_counts = greedy_coin_change(coins_us, amount)
    greedy_total = sum(greedy_counts)
    dp_min, dp_counts = dp_coin_change_min_coins(coins_us, amount)
    print(f"US coins {coins_us}, amount={amount}")
    print(" Greedy:", counts_to_str(coins_us, greedy_counts), f"(total coins={greedy_total})")
    print(" Optimal:", counts_to_str(coins_us, dp_counts), f"(min coins={dp_min})")

    # Non-canonical coin system where greedy fails
    coins_bad = [10, 6, 1]  # greedy will pick a 10 then three 1s for amount 12 => 4 coins
    amount2 = 12
    greedy_counts2 = greedy_coin_change(coins_bad, amount2)
    greedy_total2 = sum(greedy_counts2)
    dp_min2, dp_counts2 = dp_coin_change_min_coins(sorted(coins_bad, reverse=True), amount2)
    print(f"\nNon-canonical coins {coins_bad}, amount={amount2}")
    print(" Greedy:", counts_to_str(sorted(coins_bad, reverse=True), greedy_counts2), f"(total coins={greedy_total2})")
    print(" Optimal:", counts_to_str(sorted(coins_bad, reverse=True), dp_counts2), f"(min coins={dp_min2})")
