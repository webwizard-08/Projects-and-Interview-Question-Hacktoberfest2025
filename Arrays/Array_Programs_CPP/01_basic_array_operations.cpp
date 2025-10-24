//Program to demonstrate basic array operations:
//finding largest, smallest and reversing the array
#include<bits/stdc++.h>
using namespace std;

//Function to find largest element
int largestElement(vector<int> &arr){
    int n=arr.size();
    int max_element=arr[0];
    for(int i=1;i<n;i++){
        max_element=max(max_element, arr[i]);
    }
    return max_element;
}

//Function to find smallest element 
int smallestElement(vector<int> &arr){
    int n=arr.size();
    int min_element=arr[0];
    for(int i=1;i<n;i++){
        min_element=min(min_element, arr[i]);
    }
    return min_element;
}

//Function to reverse array
void reverseArray(vector<int> &arr){
    int n=arr.size();
    int left=0,right=n-1;
    while(left<right){
        swap(arr[left], arr[right]);
        left++;
        right--;
    }
}

int main(){
    vector<int> arr={3, 5, 1, 2, 4};

    cout<<"\nOriginal Array: ";
    for(auto x: arr){
        cout<<x<<" ";
    }
    cout<<endl;
    cout<<"Largest Element: "<<largestElement(arr)<<endl;
    cout<<"Smallest Element: "<<smallestElement(arr)<<endl;

    reverseArray(arr);
    cout<<"Reversed Array: ";
    for(auto x: arr){
        cout<<x<<" ";
    }
    cout<<endl;

    return 0;
}