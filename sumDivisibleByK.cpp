class Solution {
public:
    int sumDivisibleByK(vector<int>& nums, int k) {
        map<int , int>mp;
        for(int i = 0 ; i<nums.size();i++){
            mp[nums[i]]++;
        }
    int sum  = 0;
        for(auto &p : mp){
            if(p.second%k==0){
               sum =sum + p.first *p.second; 
            }
        }
        return sum;
    }
};
