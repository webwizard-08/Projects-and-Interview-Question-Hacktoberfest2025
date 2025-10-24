//Program to rotate an array to left or right by a given number of positions
//Demonstrates the use of reverse algorithm for array manipulation.

#include<bits/stdc++.h>
using namespace std;

//Rotate array to the left by d positions
void rotateArrayLeft(vector<int> &arr, int d){
    int n=arr.size();
    if(n==0) return;
    d=d%n;
    reverse(arr.begin(), arr.begin() + d);
    reverse(arr.begin() + d, arr.end());
    reverse(arr.begin(), arr.end());

    cout<<"\nLeft Rotated by "<<d<<" : ";
    for(auto x: arr){
        cout<<x<<" ";
    }
    cout<<endl;
}

//Rotate array to the right by d positions
void rotateArrayRight(vector<int> &arr, int d){
    int n=arr.size();
    if(n==0) return;
    d=d%n;
    reverse(arr.end() - d, arr.end());
    reverse(arr.begin(), arr.end() - d);
    reverse(arr.begin(), arr.end());
    
    cout<<"Right Rotated by "<<d<<" : ";
    for(auto x: arr){
        cout<<x<<" ";
    }
    cout<<endl;
}

int main(){
    vector<int> arr= {1, 2, 3, 4, 5};
    vector<int> arr2=arr;

    cout<<"Original Array: ";
    for(auto x: arr) cout<<x<<" ";
    cout<<endl;

    rotateArrayLeft(arr, 2);
    rotateArrayRight(arr2, 2);

    return 0;
}