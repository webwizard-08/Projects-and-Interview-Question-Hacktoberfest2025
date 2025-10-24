//Program to demonstrate Linear and Binary Search on arrays
#include<bits/stdc++.h>
using namespace std;

// Linear Search: O(n) time complexity
bool linearSearch(vector<int> &arr, int target){
    int n=arr.size();
    for(int i=0;i<n;i++){
        if(arr[i]==target){
            return true;
        }
    }
    return false;
}

// Binary Search (for Sorted arrays): O(log n) time complexity
bool binarySearch(vector<int> &arr, int target){
    int n=arr.size();
    int low=0, high=n-1;

    while(low<=high){
        int mid= low+ (high-low)/2;
        if(arr[mid]==target)
            return true;
        else if(arr[mid]<target)
            low=mid+1;
        else 
            high=mid-1;
    }
    return false;
}

int main(){
    vector<int> arr= {2, 3, 4, 5, 6, 7, 9};
    int target=5;

    cout<<"Linear Search for "<<target<<" : "
    <<(linearSearch(arr, target) ? "Found" : "Not Found") <<endl;

    cout<<"Binary Search for "<<target<<" : "
    <<(binarySearch(arr, target) ? "Found" : "Not Found")<<endl;
    return 0;
}