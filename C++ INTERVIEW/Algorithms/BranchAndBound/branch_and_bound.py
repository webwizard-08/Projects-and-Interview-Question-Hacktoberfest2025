from queue import Queue

# Define an Item class to represent
# each item in the knapsack
class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

# Define a Node class to represent each
# node in the branch and bound tree
class Node:
    def __init__(self, level, profit, bound, weight):
        self.level = level
        self.profit = profit
        self.bound = bound
        self.weight = weight

# Define a function to calculate the
# maximum possible profit for a given node
def bound(u, n, W, arr):
    # If the node exceeds the knapsack's
    # capacity, its profit bound is 0
    if u.weight >= W:
        return 0

    # Calculate the profit bound by adding
    # the profits of all remaining items
    # that can fit into the knapsack
    profitBound = u.profit
    j = u.level + 1
    totWeight = u.weight

    while j < n and totWeight + arr[j].weight <= W:
        totWeight += arr[j].weight
        profitBound += arr[j].value
        j += 1

    # If there are still items remaining,
    # add a fraction of the next item's
    # profit proportional to the remaining
    # space in the knapsack
    if j < n:
        profitBound += (W - totWeight) * arr[j].value / arr[j].weight

    return profitBound

# Define the knapsack_solution function
# that uses the Branch and Bound algorithm
# to solve the 0-1 Knapsack problem
def knapsack_solution(W, arr, n):

    # Sort the items in descending order
    # of their value-to-weight ratio
    arr.sort(key=lambda item: item.value / item.weight, reverse=True)

    # Initialize a queue with a root node
    # of the branch and bound tree
    q = Queue()
    u = Node(-1, 0, 0, 0)
    q.put(u)

    # Initialize a variable to keep track
    # of the maximum profit found so far
    maxProfit = 0

    # Loop through each node in the
    # branch and bound tree
    while not q.empty():
        u = q.get()

        # If the node is a leaf node, skip it
        if u.level == n - 1:
            continue

        # Calculate the child node that
        # includes the next item in knapsack
        v_level = u.level + 1
        v_profit = u.profit + arr[v_level].value
        v_weight = u.weight + arr[v_level].weight
        v = Node(v_level, v_profit, 0, v_weight)

        # If the child node's weight is
        # less than or equal to the knapsack's
        # capacity and its profit is greater
        # than the maximum profit found so far,
        # update the maximum profit
        if v.weight <= W and v.profit > maxProfit:
            maxProfit = v.profit

        # Calculate the profit bound for the
        # child node and add it to the queue
        # if its profit bound is greater than
        # the maximum profit found so far
        v.bound = bound(v, n, W, arr)

        if v.bound > maxProfit:
            q.put(v)

        # Do the same thing, but Without taking
        # the item in knapsack
        v_level = u.level + 1
        v_profit = u.profit
        v_weight = u.weight
        v = Node(v_level, v_profit, 0, v_weight)
        
        v.bound = bound(v, n, W, arr)

        if v.bound > maxProfit:
            q.put(v)

    return maxProfit


# Driver Code
if __name__ == '__main__':
    W = 10
    arr = [Item(2, 40), Item(3.14, 50), Item(
        1.98, 100), Item(5, 95), Item(3, 30)]
    n = len(arr)

    print('Maximum possible profit =', knapsack_solution(W, arr, n))
