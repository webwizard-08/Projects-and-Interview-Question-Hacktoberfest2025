Time Limit : 90 min
2 Questions

Problem 1

You want to schedule a certain number of trips with a collection of several cabs.

Given an integer n representing a desired number of trips, and an array cabTravelTime representing your cabs and how long it takes each cab (at that index of the array) to make a trip, return the minimum time required to make n trips.

Assume that cabs can run simultaneously and there is no waiting period between trips. There may be multiple cabs with the same time cost.*

Examples
If n=3 and cabTravelTime=[1,2], then the answer is 2. This is because the first cab (index 0, cost 1) can make 2 trips costing a total of 2 time units, and the second cab can make a single trip costing 2 at the same time.

n=10
cabTravelTime=[1,3,5,7,8]

* 7 trips with cab 0 (cost 1)
* 2 trips with cab 1 (cost 3)
* 1 trip with cab 2 (cost 5)
So, answer is 7 (there could be other combinations)

n=3
cabTravelTime=[3,4,8]

* 2 trips with cab 0 (cost 6)
* 1 trip with cab 1 (cost 4)
Time = 6


Problem 2

There is a long road with markers on it after each unit of distance. There are some ubers standing on the road. You are given the starting and ending coordinate of each uber (both inclusive).
Note: At any given marker there may be multiple ubers or there may be none at all.

Your task is to find the number of markers on which at least one uber is present. An uber with coordinates (l, r) is considered to be present on a marker m if and only if l ≤ m ≤ r.

Example

For coordinates=[[4, 7], [-1, 5], [3, 6]], the output should be easyCountUber(coordinates) = 9.

solutions:- 
Q1&Q2 Python 3
Q1 BS NlogN

from typing import List

def min_cost(ntrips, cab_costs: List[int]):
    cab_costs.sort()
    lo, hi = 0, ntrips*cab_costs[0]    # low and high answer choice
    while lo < hi:
        mid = (lo+hi)//2   
        # calculate max number of trips possible using smallest cab costs <= mid cost
        cnt_trips = 0
        for cc in cab_costs:
            cnt_trips += mid//cc if mid >= cc else 0
            if cc > mid or cnt_trips >= ntrips:
                break

        if cnt_trips >= ntrips:
            hi = mid
        else:
            lo = mid+1
    return lo

if __name__ == '__main__':
    print(min_cost(10,[1,3,5,7,8]))  # 7
    print(min_cost(3,[3,4,8]))  # 6
  
Q2 N interval merge

from typing import List


def cab_coverage(interv: List[List]):
    markerCnt = lambda x,y: y-x+1
    interv.sort()
    res = None
    ans = 0
    for i,(l,r) in enumerate(interv):
        if not res:
            res = [l,r]
            continue

        prevl, prevr = res
        if l<=prevr:
            res[1] = r
        else:
            ans += markerCnt(res[0], res[1])
            res = [l,r]
    return ans + markerCnt(res[0], res[1])


if __name__ == '__main__':
    print(cab_coverage([[4,7],[-1,5],[3,6]]))  # 9
