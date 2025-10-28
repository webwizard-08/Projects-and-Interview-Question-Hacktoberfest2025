/**
 * Capacity to Ship Packages Within D Days
 * 
 * Problem: A conveyor belt has packages that must be shipped from one port to another within D days.
 * The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with 
 * packages in the order given by weights. We may not load more weight than the maximum weight capacity of the ship.
 * 
 * Return the least weight capacity of the ship that will result in all the packages being shipped within D days.
 * 
 * Example 1:
 * Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
 * Output: 15
 * Explanation: A ship capacity of 15 is the minimum to ship all packages in 5 days:
 * Day 1: 1, 2, 3, 4, 5
 * Day 2: 6, 7
 * Day 3: 8
 * Day 4: 9
 * Day 5: 10
 * 
 * Example 2:
 * Input: weights = [3,2,2,4,1,4], days = 3
 * Output: 6
 * Explanation: A ship capacity of 6 is the minimum:
 * Day 1: 3, 2
 * Day 2: 2, 4
 * Day 3: 1, 4
 * 
 * Constraints:
 * - 1 <= days <= weights.length <= 50000
 * - 1 <= weights[i] <= 500
 */

public class ShipPackages {
    
    /**
     * Find the minimum ship capacity to ship all packages within D days
     * Time Complexity: O(n * log(sum - max)) where n = weights.length
     * Space Complexity: O(1)
     * 
     * Approach: Binary Search on Answer
     * - The minimum possible capacity is max(weights) - we must be able to carry the heaviest package
     * - The maximum possible capacity is sum(weights) - ship everything in one day
     * - We binary search in this range to find the minimum capacity that works
     */
    public static int shipWithinDays(int[] weights, int days) {
        // Edge case
        if (weights == null || weights.length == 0 || days <= 0) {
            return 0;
        }
        
        // Calculate search space bounds
        int maxWeight = 0;      // Minimum possible capacity
        int totalWeight = 0;    // Maximum possible capacity
        
        for (int weight : weights) {
            maxWeight = Math.max(maxWeight, weight);
            totalWeight += weight;
        }
        
        // Binary search for minimum capacity
        int left = maxWeight;
        int right = totalWeight;
        int result = totalWeight;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            // Check if we can ship all packages with capacity 'mid' in 'days' days
            if (canShip(weights, days, mid)) {
                result = mid;           // Valid capacity, try to minimize
                right = mid - 1;
            } else {
                left = mid + 1;         // Capacity too small, increase
            }
        }
        
        return result;
    }
    
    /**
     * Helper function to check if all packages can be shipped with given capacity
     * Time Complexity: O(n)
     */
    private static boolean canShip(int[] weights, int days, int capacity) {
        int daysNeeded = 1;
        int currentLoad = 0;
        
        for (int weight : weights) {
            // If adding this package exceeds capacity, need a new day
            if (currentLoad + weight > capacity) {
                daysNeeded++;
                currentLoad = weight;
                
                // If we've exceeded available days, this capacity doesn't work
                if (daysNeeded > days) {
                    return false;
                }
            } else {
                currentLoad += weight;
            }
        }
        
        return true;
    }
    
    /**
     * Alternative approach: More verbose but clearer logic
     */
    public static int shipWithinDaysVerbose(int[] weights, int days) {
        int minCapacity = 0;
        int maxCapacity = 0;
        
        // Find the range for binary search
        for (int weight : weights) {
            minCapacity = Math.max(minCapacity, weight);  // Must carry heaviest package
            maxCapacity += weight;                         // Could carry all at once
        }
        
        // Binary search for minimum valid capacity
        while (minCapacity < maxCapacity) {
            int midCapacity = minCapacity + (maxCapacity - minCapacity) / 2;
            
            // Calculate how many days needed with this capacity
            int daysRequired = calculateDaysNeeded(weights, midCapacity);
            
            if (daysRequired <= days) {
                // This capacity works, try smaller
                maxCapacity = midCapacity;
            } else {
                // This capacity is too small, try larger
                minCapacity = midCapacity + 1;
            }
        }
        
        return minCapacity;
    }
    
    /**
     * Calculate how many days are needed with given ship capacity
     */
    private static int calculateDaysNeeded(int[] weights, int capacity) {
        int days = 1;
        int currentWeight = 0;
        
        for (int weight : weights) {
            if (currentWeight + weight > capacity) {
                days++;
                currentWeight = weight;
            } else {
                currentWeight += weight;
            }
        }
        
        return days;
    }
    
    /**
     * Main method with test cases
     */
    public static void main(String[] args) {
        System.out.println("=== Capacity to Ship Packages Within D Days ===\n");
        
        // Test Case 1
        int[] weights1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        int days1 = 5;
        System.out.println("Test Case 1:");
        System.out.println("Weights: " + arrayToString(weights1));
        System.out.println("Days: " + days1);
        System.out.println("Minimum Capacity: " + shipWithinDays(weights1, days1));
        System.out.println("Expected: 15\n");
        
        // Test Case 2
        int[] weights2 = {3, 2, 2, 4, 1, 4};
        int days2 = 3;
        System.out.println("Test Case 2:");
        System.out.println("Weights: " + arrayToString(weights2));
        System.out.println("Days: " + days2);
        System.out.println("Minimum Capacity: " + shipWithinDays(weights2, days2));
        System.out.println("Expected: 6\n");
        
        // Test Case 3: All packages same weight
        int[] weights3 = {10, 10, 10, 10, 10};
        int days3 = 2;
        System.out.println("Test Case 3:");
        System.out.println("Weights: " + arrayToString(weights3));
        System.out.println("Days: " + days3);
        System.out.println("Minimum Capacity: " + shipWithinDays(weights3, days3));
        System.out.println("Expected: 30\n");
        
        // Test Case 4: One day only
        int[] weights4 = {1, 2, 3, 4, 5};
        int days4 = 1;
        System.out.println("Test Case 4:");
        System.out.println("Weights: " + arrayToString(weights4));
        System.out.println("Days: " + days4);
        System.out.println("Minimum Capacity: " + shipWithinDays(weights4, days4));
        System.out.println("Expected: 15 (sum of all)\n");
        
        // Test Case 5: Days equal to packages
        int[] weights5 = {5, 10, 15, 20};
        int days5 = 4;
        System.out.println("Test Case 5:");
        System.out.println("Weights: " + arrayToString(weights5));
        System.out.println("Days: " + days5);
        System.out.println("Minimum Capacity: " + shipWithinDays(weights5, days5));
        System.out.println("Expected: 20 (max weight)\n");
        
        // Test Case 6: Large single package
        int[] weights6 = {1, 1, 1, 100, 1, 1, 1};
        int days6 = 3;
        System.out.println("Test Case 6:");
        System.out.println("Weights: " + arrayToString(weights6));
        System.out.println("Days: " + days6);
        System.out.println("Minimum Capacity: " + shipWithinDays(weights6, days6));
        System.out.println("Expected: 100\n");
        
        // Verbose method comparison
        System.out.println("=== Verbose Method Test ===");
        System.out.println("Weights: " + arrayToString(weights1));
        System.out.println("Days: " + days1);
        System.out.println("Result (Verbose): " + shipWithinDaysVerbose(weights1, days1));
    }
    
    /**
     * Helper method to convert array to string
     */
    private static String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}