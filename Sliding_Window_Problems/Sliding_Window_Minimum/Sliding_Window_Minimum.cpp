// Problem Link :- https://cses.fi/problemset/task/3221

#include<bits/stdc++.h>
using namespace std;
 
void solve()
{
    long long n, k;
    cin>>n>>k;
    long long x, a, b, c;
    cin>>x>>a>>b>>c;
    deque<pair<long long, long long>> dq;
    // vector<long long> arr{x};
    long long prev = x;
    long long curr = 0;
    long long mini = x;
    dq.push_front({x, 0});
    for(long long i=1;i<k;i++)
    {
        curr = (a*prev + b)%c;
        prev = curr;
        mini = min(mini, curr);
        if(dq.front().first >= curr)
        {
            dq.clear();
        }
        else
        {
            while(!dq.empty())
            {
                pair<long long, long long> p = dq.back();
                if(p.first >= curr)
                {
                    dq.pop_back();
                }
                else
                {
                    break;
                }
            }
        }
        dq.push_back({curr, i});
        // arr.push_back(curr);
    }
    long long ans = mini;
    long long r = k;
    long long l = 1;
    while(r<n)
    {
        curr = (a*prev + b)%c;
        prev = curr;
        while(!dq.empty())
        {
            pair<long long, long long> p = dq.front();
            if(p.second < l)
                dq.pop_front();
            else
                break;
        }
        if(dq.front().first >= curr)
        {
            dq.clear();
        }
        else
        {
            while(!dq.empty())
            {
                pair<long long, long long> p = dq.back();
                if(p.first >= curr)
                {
                    dq.pop_back();
                }
                else
                {
                    break;
                }
            }
        }
        dq.push_back({curr, r});
        mini = dq.front().first;
        l++;
        r++;
        ans = ans^mini;
    }
    cout<<ans<<endl;
}
 
int main()
{
    long long t = 1;
    // cin>>t;
    while(t--)
    {
        solve();
    }
}
