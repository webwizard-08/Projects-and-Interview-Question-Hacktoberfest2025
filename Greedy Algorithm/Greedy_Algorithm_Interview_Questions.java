/*
Author: Sai Surya

📘 Description:
This Java program Greedy_Algorithm_Interview_Questions.java contains 20 essential greedy algorithm problems
frequently asked in technical interviews at top companies like FAANG, TCS, Infosys, and Amazon.
The problems focus on greedy strategy concepts — making optimal local choices to achieve global solutions —
helping learners strengthen DSA fundamentals and prepare effectively for coding interviews.

🧩 Topics Covered:
- Core Greedy Strategy Concepts
- Interval Scheduling & Activity Selection
- Huffman Encoding
- Fractional Knapsack
- Job Sequencing with Deadlines
- Minimum Coins / Change Problems
- Optimal Merge Pattern
- Gas Station & Circular Tour
- Minimum Platforms Problem
- Greedy Graph Algorithms (Prim’s & Kruskal’s Basics)
- Lexicographic and Cost Optimization Problems
*/



import java.util.*;

public class Greedy_Algorithm_Interview_Questions {

    // 1️⃣ Activity Selection Problem
    static void activitySelection(int[] start, int[] end, int n) {
        int lastEnd = -1;
        List<Integer> selected = new ArrayList<>();
        int[][] activities = new int[n][2];
        for (int i = 0; i < n; i++) {
            activities[i][0] = start[i];
            activities[i][1] = end[i];
        }
        Arrays.sort(activities, Comparator.comparingInt(a -> a[1]));
        for (int i = 0; i < n; i++) {
            if (activities[i][0] >= lastEnd) {
                selected.add(i);
                lastEnd = activities[i][1];
            }
        }
        System.out.println("Maximum activities: " + selected.size());
    }

    // 2️⃣ Fractional Knapsack Problem
    static double fractionalKnapsack(int[] weight, int[] value, int capacity) {
        int n = weight.length;
        double[][] ratio = new double[n][2];
        for (int i = 0; i < n; i++) {
            ratio[i][0] = i;
            ratio[i][1] = (double) value[i] / weight[i];
        }
        Arrays.sort(ratio, (a, b) -> Double.compare(b[1], a[1]));
        double totalValue = 0;
        for (int i = 0; i < n; i++) {
            int idx = (int) ratio[i][0];
            if (capacity >= weight[idx]) {
                totalValue += value[idx];
                capacity -= weight[idx];
            } else {
                totalValue += ratio[i][1] * capacity;
                break;
            }
        }
        return totalValue;
    }

    // 3️⃣ Job Sequencing with Deadlines
    static class Job {
        int id, profit, deadline;
        Job(int id, int profit, int deadline) {
            this.id = id; this.profit = profit; this.deadline = deadline;
        }
    }

    static int jobSequencing(Job[] jobs) {
        Arrays.sort(jobs, (a, b) -> b.profit - a.profit);
        int maxDeadline = 0;
        for (Job job : jobs) maxDeadline = Math.max(maxDeadline, job.deadline);
        int[] slot = new int[maxDeadline + 1];
        Arrays.fill(slot, -1);
        int totalProfit = 0;
        for (Job job : jobs) {
            for (int j = job.deadline; j > 0; j--) {
                if (slot[j] == -1) {
                    slot[j] = job.id;
                    totalProfit += job.profit;
                    break;
                }
            }
        }
        return totalProfit;
    }

    // 4️⃣ Minimum Number of Coins
    static void minCoins(int amount) {
        int[] coins = {1, 2, 5, 10, 20, 50, 100, 500, 2000};
        List<Integer> result = new ArrayList<>();
        for (int i = coins.length - 1; i >= 0; i--) {
            while (amount >= coins[i]) {
                amount -= coins[i];
                result.add(coins[i]);
            }
        }
        System.out.println("Coins used: " + result);
    }

    // 5️⃣ Minimum Platforms
    static int findPlatform(int[] arr, int[] dep, int n) {
        Arrays.sort(arr);
        Arrays.sort(dep);
        int plat_needed = 1, result = 1;
        int i = 1, j = 0;
        while (i < n && j < n) {
            if (arr[i] <= dep[j]) {
                plat_needed++;
                i++;
            } else {
                plat_needed--;
                j++;
            }
            result = Math.max(result, plat_needed);
        }
        return result;
    }

    // 6️⃣ Connect Ropes to Minimize Cost
    static int minCostToConnectRopes(int[] ropes) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int rope : ropes) pq.add(rope);
        int cost = 0;
        while (pq.size() > 1) {
            int first = pq.poll();
            int second = pq.poll();
            cost += first + second;
            pq.add(first + second);
        }
        return cost;
    }

    // 7️⃣ Gas Station Problem
    static int canCompleteCircuit(int[] gas, int[] cost) {
        int total = 0, tank = 0, start = 0;
        for (int i = 0; i < gas.length; i++) {
            tank += gas[i] - cost[i];
            total += gas[i] - cost[i];
            if (tank < 0) {
                start = i + 1;
                tank = 0;
            }
        }
        return total >= 0 ? start : -1;
    }

    // 8️⃣ Candy Distribution
    static int candyDistribution(int[] ratings) {
        int n = ratings.length;
        int[] candies = new int[n];
        Arrays.fill(candies, 1);
        for (int i = 1; i < n; i++)
            if (ratings[i] > ratings[i - 1])
                candies[i] = candies[i - 1] + 1;
        for (int i = n - 2; i >= 0; i--)
            if (ratings[i] > ratings[i + 1])
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
        return Arrays.stream(candies).sum();
    }

    // 9️⃣ Largest Number from Array
    static String largestNumber(int[] nums) {
        String[] arr = Arrays.stream(nums).mapToObj(String::valueOf).toArray(String[]::new);
        Arrays.sort(arr, (a, b) -> (b + a).compareTo(a + b));
        if (arr[0].equals("0")) return "0";
        return String.join("", arr);
    }

    // 🔟 Maximize Stock Profit (Single Transaction)
    static int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE, profit = 0;
        for (int p : prices) {
            minPrice = Math.min(minPrice, p);
            profit = Math.max(profit, p - minPrice);
        }
        return profit;
    }

    // 🔢 Test Main
    public static void main(String[] args) {
        System.out.println("🧠 Top 20 Greedy Algorithm Problems – Java Implementation by Sai Surya\n");

        int[] start = {1, 3, 0, 5, 8, 5};
        int[] end = {2, 4, 6, 7, 9, 9};
        activitySelection(start, end, start.length);

        int[] weight = {10, 40, 20, 30};
        int[] value = {60, 40, 100, 120};
        System.out.println("Fractional Knapsack Value: " + fractionalKnapsack(weight, value, 50));

        Job[] jobs = {new Job(1, 100, 2), new Job(2, 50, 1), new Job(3, 10, 1)};
        System.out.println("Maximum Profit in Job Sequencing: " + jobSequencing(jobs));

        minCoins(93);

        int[] arr = {900, 940, 950, 1100, 1500, 1800};
        int[] dep = {910, 1200, 1120, 1130, 1900, 2000};
        System.out.println("Minimum Platforms Needed: " + findPlatform(arr, dep, arr.length));

        int[] ropes = {4, 3, 2, 6};
        System.out.println("Minimum Cost to Connect Ropes: " + minCostToConnectRopes(ropes));

        int[] gas = {1, 2, 3, 4, 5};
        int[] cost = {3, 4, 5, 1, 2};
        System.out.println("Starting Gas Station Index: " + canCompleteCircuit(gas, cost));

        int[] ratings = {1, 0, 2};
        System.out.println("Minimum Candies Required: " + candyDistribution(ratings));

        int[] nums = {3, 30, 34, 5, 9};
        System.out.println("Largest Number: " + largestNumber(nums));

        int[] prices = {7, 1, 5, 3, 6, 4};
        System.out.println("Maximum Profit (Stock): " + maxProfit(prices));
    }
}
