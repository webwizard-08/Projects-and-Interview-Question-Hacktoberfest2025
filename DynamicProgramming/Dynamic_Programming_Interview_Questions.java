// Author: Sai Surya
// File: Dynamic_Programming_Interview_Questions.java
// 📘 20 Dynamic Programming Problems with Java Solutions
import java.util.*;
public class Dynamic_Programming_Interview_Questions {
    
    // 1️⃣ Fibonacci (Top-Down + Memoization)
    static int fibMemo(int n, int[] dp) {
        if (n <= 1) return n;
        if (dp[n] != -1) return dp[n];
        return dp[n] = fibMemo(n - 1, dp) + fibMemo(n - 2, dp);
    }

    // 2️⃣ Climbing Stairs
    static int climbStairs(int n) {
        if (n <= 2) return n;
        int[] dp = new int[n + 1];
        dp[1] = 1; dp[2] = 2;
        for (int i = 3; i <= n; i++) dp[i] = dp[i - 1] + dp[i - 2];
        return dp[n];
    }

    // 3️⃣ 0/1 Knapsack
    static int knapsack(int[] wt, int[] val, int W, int n) {
        int[][] dp = new int[n + 1][W + 1];
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= W; w++) {
                if (wt[i - 1] <= w)
                    dp[i][w] = Math.max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
                else
                    dp[i][w] = dp[i - 1][w];
            }
        }
        return dp[n][W];
    }

    // 4️⃣ Subset Sum
    static boolean subsetSum(int[] arr, int sum) {
        int n = arr.length;
        boolean[][] dp = new boolean[n + 1][sum + 1];
        for (int i = 0; i <= n; i++) dp[i][0] = true;
        for (int i = 1; i <= n; i++) {
            for (int s = 1; s <= sum; s++) {
                if (arr[i - 1] <= s)
                    dp[i][s] = dp[i - 1][s - arr[i - 1]] || dp[i - 1][s];
                else dp[i][s] = dp[i - 1][s];
            }
        }
        return dp[n][sum];
    }

    // 5️⃣ Equal Partition Sum
    static boolean equalPartition(int[] arr) {
        int sum = Arrays.stream(arr).sum();
        if (sum % 2 != 0) return false;
        return subsetSum(arr, sum / 2);
    }

    // 6️⃣ Count of Subsets with Given Sum
    static int countSubsets(int[] arr, int sum) {
        int n = arr.length;
        int[][] dp = new int[n + 1][sum + 1];
        for (int i = 0; i <= n; i++) dp[i][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int s = 1; s <= sum; s++) {
                if (arr[i - 1] <= s)
                    dp[i][s] = dp[i - 1][s - arr[i - 1]] + dp[i - 1][s];
                else dp[i][s] = dp[i - 1][s];
            }
        }
        return dp[n][sum];
    }

    // 7️⃣ Minimum Subset Sum Difference
    static int minSubsetDiff(int[] arr) {
        int n = arr.length, sum = Arrays.stream(arr).sum();
        boolean[][] dp = new boolean[n + 1][sum / 2 + 1];
        for (int i = 0; i <= n; i++) dp[i][0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= sum / 2; j++) {
                if (arr[i - 1] <= j)
                    dp[i][j] = dp[i - 1][j - arr[i - 1]] || dp[i - 1][j];
                else dp[i][j] = dp[i - 1][j];
            }
        }
        int diff = Integer.MAX_VALUE;
        for (int j = sum / 2; j >= 0; j--) {
            if (dp[n][j]) {
                diff = sum - 2 * j;
                break;
            }
        }
        return diff;
    }

    // 8️⃣ Coin Change (Min Coins)
    static int coinChange(int[] coins, int sum) {
        int[] dp = new int[sum + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= sum; i++) {
            for (int c : coins) {
                if (i >= c && dp[i - c] != Integer.MAX_VALUE)
                    dp[i] = Math.min(dp[i], dp[i - c] + 1);
            }
        }
        return dp[sum] == Integer.MAX_VALUE ? -1 : dp[sum];
    }

    // 9️⃣ Longest Common Subsequence
    static int lcs(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = (s1.charAt(i - 1) == s2.charAt(j - 1))
                        ? 1 + dp[i - 1][j - 1]
                        : Math.max(dp[i - 1][j], dp[i][j - 1]);
        return dp[n][m];
    }

    // 🔟 Longest Common Substring
    static int longestCommonSubstring(String s1, String s2) {
        int n = s1.length(), m = s2.length(), ans = 0;
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    ans = Math.max(ans, dp[i][j]);
                }
        return ans;
    }

    // 11️⃣ Longest Palindromic Subsequence
    static int longestPalSubseq(String s) {
        String rev = new StringBuilder(s).reverse().toString();
        return lcs(s, rev);
    }

    // 12️⃣ Edit Distance
    static int editDistance(String s1, String s2) {
        int n = s1.length(), m = s2.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) dp[i][0] = i;
        for (int j = 0; j <= m; j++) dp[0][j] = j;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = (s1.charAt(i - 1) == s2.charAt(j - 1))
                        ? dp[i - 1][j - 1]
                        : 1 + Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1]));
        return dp[n][m];
    }

    // 13️⃣ Matrix Chain Multiplication
    static int matrixChain(int[] arr, int i, int j, int[][] dp) {
        if (i == j) return 0;
        if (dp[i][j] != -1) return dp[i][j];
        int min = Integer.MAX_VALUE;
        for (int k = i; k < j; k++) {
            int cost = matrixChain(arr, i, k, dp)
                    + matrixChain(arr, k + 1, j, dp)
                    + arr[i - 1] * arr[k] * arr[j];
            min = Math.min(min, cost);
        }
        return dp[i][j] = min;
    }

    // 14️⃣ Minimum Cost Path in Matrix
    static int minPathSum(int[][] grid) {
        int n = grid.length, m = grid[0].length;
        int[][] dp = new int[n][m];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) dp[i][0] = dp[i - 1][0] + grid[i][0];
        for (int j = 1; j < m; j++) dp[0][j] = dp[0][j - 1] + grid[0][j];
        for (int i = 1; i < n; i++)
            for (int j = 1; j < m; j++)
                dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j], dp[i][j - 1]);
        return dp[n - 1][m - 1];
    }

    // 15️⃣ Maximum Product Subarray
    static int maxProduct(int[] nums) {
        int maxProd = nums[0], minProd = nums[0], res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < 0) {
                int tmp = maxProd;
                maxProd = minProd;
                minProd = tmp;
            }
            maxProd = Math.max(nums[i], nums[i] * maxProd);
            minProd = Math.min(nums[i], nums[i] * minProd);
            res = Math.max(res, maxProd);
        }
        return res;
    }

    // 16️⃣ Rod Cutting Problem
    static int rodCutting(int[] price, int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++)
            for (int j = 0; j < i; j++)
                dp[i] = Math.max(dp[i], price[j] + dp[i - j - 1]);
        return dp[n];
    }

    // 17️⃣ Palindrome Partitioning (Min Cuts)
    static int minCutPalindrome(String s) {
        int n = s.length();
        boolean[][] pal = new boolean[n][n];
        int[] cuts = new int[n];
        for (int i = 0; i < n; i++) {
            int minCut = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(i) == s.charAt(j) && (i - j <= 1 || pal[j + 1][i - 1])) {
                    pal[j][i] = true;
                    minCut = (j == 0) ? 0 : Math.min(minCut, cuts[j - 1] + 1);
                }
            }
            cuts[i] = minCut;
        }
        return cuts[n - 1];
    }

    // 18️⃣ House Robber Problem
    static int houseRobber(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++)
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        return dp[nums.length - 1];
    }

    // 19️⃣ Decode Ways
    static int numDecodings(String s) {
        if (s.charAt(0) == '0') return 0;
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1; dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            if (s.charAt(i - 1) != '0') dp[i] += dp[i - 1];
            int two = Integer.parseInt(s.substring(i - 2, i));
            if (two >= 10 && two <= 26) dp[i] += dp[i - 2];
        }
        return dp[n];
    }

    // 20️⃣ Egg Dropping Puzzle
    static int eggDrop(int e, int f) {
        int[][] dp = new int[e + 1][f + 1];
        for (int i = 1; i <= e; i++) {
            dp[i][0] = 0;
            dp[i][1] = 1;
        }
        for (int j = 1; j <= f; j++) dp[1][j] = j;
        for (int i = 2; i <= e; i++)
            for (int j = 2; j <= f; j++) {
                dp[i][j] = Integer.MAX_VALUE;
                for (int x = 1; x <= j; x++) {
                    int res = 1 + Math.max(dp[i - 1][x - 1], dp[i][j - x]);
                    dp[i][j] = Math.min(dp[i][j], res);
                }
            }
        return dp[e][f];
    }

    // 🧪 Main Function
    public static void main(String[] args) {
        int[] dp = new int[10];
        Arrays.fill(dp, -1);
        System.out.println("1. Fibonacci(6): " + fibMemo(6, dp));
        System.out.println("2. Climb Stairs(5): " + climbStairs(5));
        System.out.println("3. Knapsack: " + knapsack(new int[]{1, 3, 4, 5}, new int[]{1, 4, 5, 7}, 7, 4));
        System.out.println("4. Subset Sum(11): " + subsetSum(new int[]{2, 3, 7, 8, 10}, 11));
        System.out.println("5. Equal Partition: " + equalPartition(new int[]{1, 5, 11, 5}));
        System.out.println("6. Count Subsets(10): " + countSubsets(new int[]{2, 3, 5, 6, 8, 10}, 10));
        System.out.println("7. Min Subset Diff: " + minSubsetDiff(new int[]{1, 6, 11, 5}));
        System.out.println("8. Coin Change: " + coinChange(new int[]{1, 2, 5}, 11));
        System.out.println("9. LCS: " + lcs("abcde", "ace"));
        System.out.println("10. LCSUBSTRING: " + longestCommonSubstring("abcdxyz", "xyzabcd"));
        System.out.println("11. Longest Pal Subseq: " + longestPalSubseq("bbbab"));
        System.out.println("12. Edit Distance: " + editDistance("horse", "ros"));
        int[][] mcmDP = new int[5][5];
        for (int[] row : mcmDP) Arrays.fill(row, -1);
        System.out.println("13. MCM: " + matrixChain(new int[]{1, 2, 3, 4}, 1, 3, mcmDP));
        int[][] grid = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
        System.out.println("14. Min Path Sum: " + minPathSum(grid));
        System.out.println("15. Max Product: " + maxProduct(new int[]{2, 3, -2, 4}));
        System.out.println("16. Rod Cutting: " + rodCutting(new int[]{1, 5, 8, 9}, 4));
        System.out.println("17. Min Cut Palindrome: " + minCutPalindrome("aab"));
        System.out.println("18. House Robber: " + houseRobber(new int[]{2, 7, 9, 3, 1}));
        System.out.println("19. Decode Ways: " + numDecodings("226"));
        System.out.println("20. Egg Drop (2 eggs, 10 floors): " + eggDrop(2, 10));
    }
}
Dynamic_Programming_Interview_Questions.java
