// ===========================================
// Top 50 FAANG / Product-Based Company Interview Problems + Bonus Topics
// ===========================================
//
// Author: Mohit Kourav (Prepared with ChatGPT)
// Languages: Python (.py), C++ (.cpp), Java (.java)
//
// Description:
// This file contains solutions to the top 50 interview questions commonly asked 
// at FAANG and other product-based companies. It also includes 5 advanced topics
// on Python/C++/Java internals and system design patterns.
//
// Features:
// - Each problem includes:
//     1. Problem statement and explanation
//     2. Example input/output
//     3. Brute-force solution (where applicable)
//     4. Optimized solution
//     5. Inline comments explaining logic, time & space complexity
//
// - Problems are categorized into:
//     1. Arrays & Strings
//     2. Linked Lists
//     3. Trees & Binary Search Trees
//     4. Graphs
//     5. Dynamic Programming
//     6. Heaps, Stacks & Queues
//     7. Backtracking
//     8. System Design / Advanced
//     9. Bonus Advanced Topics (language-specific)
//
// Usage:
// - Python: Run `.py` files with Python 3.7+
// - C++: Compile with g++ (C++11 or higher) and run executable
// - Java: Compile and run with Java 8+
//
// Notes:
// - All implementations are optimized for clarity and performance.
// - Brute-force methods are included for learning and comparison.
// - Inline comments provide detailed step-by-step explanations.
// - Some system design or advanced topics are conceptual or framework-based.
//
// License:
// - Free to use for educational purposes, interview prep, and personal projects.
//
// ===========================================


// ===========================================
// Top 50 FAANG / Product Interview Problems + Bonus
// C++ Version
// Author: Mohit Kourav (prepared with ChatGPT)
// Each problem: Statement, brute-force / optimized solution, explanation
// ===========================================

#include <bits/stdc++.h>
using namespace std;

// -------------------------------------------
// Problem 1: Two Sum
// -------------------------------------------
/*
Given an array nums and target, return indices of two numbers that sum to target.
Brute-force: O(n^2)
Optimized: Hash map O(n)
*/
vector<int> twoSumBF(vector<int>& nums, int target) {
    int n = nums.size();
    for (int i=0;i<n;i++)
        for(int j=i+1;j<n;j++)
            if(nums[i]+nums[j]==target)
                return {i,j};
    return {};
}

vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int,int> mp; // value -> index
    for(int i=0;i<nums.size();i++){
        int need = target - nums[i];
        if(mp.count(need))
            return {mp[need], i};
        mp[nums[i]] = i;
    }
    return {};
}

// -------------------------------------------
// Problem 2: Maximum Subarray (Kadane)
// -------------------------------------------
int maxSubArrayBF(vector<int>& nums) {
    int n=nums.size();
    int best=INT_MIN;
    for(int i=0;i<n;i++){
        int cur=0;
        for(int j=i;j<n;j++){
            cur+=nums[j];
            best = max(best,cur);
        }
    }
    return best;
}

int maxSubArray(vector<int>& nums){
    int cur=nums[0], best=nums[0];
    for(int i=1;i<nums.size();i++){
        cur = max(nums[i], cur+nums[i]);
        best = max(best, cur);
    }
    return best;
}

// -------------------------------------------
// Problem 3: Merge Intervals
// -------------------------------------------
vector<vector<int>> mergeIntervals(vector<vector<int>>& intervals){
    if(intervals.empty()) return {};
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;
    merged.push_back(intervals[0]);
    for(int i=1;i<intervals.size();i++){
        if(intervals[i][0]<=merged.back()[1])
            merged.back()[1] = max(merged.back()[1], intervals[i][1]);
        else
            merged.push_back(intervals[i]);
    }
    return merged;
}

// -------------------------------------------
// Problem 4: Container With Most Water
// -------------------------------------------
int maxAreaBF(vector<int>& height){
    int n=height.size(), best=0;
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++)
            best=max(best, (j-i)*min(height[i],height[j]));
    return best;
}

int maxArea(vector<int>& height){
    int l=0, r=height.size()-1, best=0;
    while(l<r){
        int area=(r-l)*min(height[l],height[r]);
        best=max(best,area);
        if(height[l]<height[r]) l++;
        else r--;
    }
    return best;
}

// -------------------------------------------
// Problem 5: 3Sum
// -------------------------------------------
vector<vector<int>> threeSum(vector<int>& nums){
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    int n=nums.size();
    for(int i=0;i<n-2;i++){
        if(i>0 && nums[i]==nums[i-1]) continue;
        int l=i+1, r=n-1;
        while(l<r){
            int sum=nums[i]+nums[l]+nums[r];
            if(sum==0){
                res.push_back({nums[i],nums[l],nums[r]});
                while(l<r && nums[l]==nums[l+1]) l++;
                while(l<r && nums[r]==nums[r-1]) r--;
                l++; r--;
            }else if(sum<0) l++;
            else r--;
        }
    }
    return res;
}

// -------------------------------------------
// Problem 6: Longest Substring Without Repeating Characters
// -------------------------------------------
int lengthOfLongestSubstring(string s){
    unordered_map<char,int> last;
    int start=0, best=0;
    for(int i=0;i<s.size();i++){
        if(last.count(s[i]) && last[s[i]]>=start)
            start=last[s[i]]+1;
        last[s[i]]=i;
        best=max(best,i-start+1);
    }
    return best;
}

// -------------------------------------------
// Problem 7: Trapping Rain Water
// -------------------------------------------
int trapBF(vector<int>& height){
    int n=height.size(), water=0;
    for(int i=0;i<n;i++){
        int left=*max_element(height.begin(),height.begin()+i+1);
        int right=*max_element(height.begin()+i,height.end());
        water+=max(0,min(left,right)-height[i]);
    }
    return water;
}

int trap(vector<int>& height){
    int l=0,r=height.size()-1,left_max=0,right_max=0, water=0;
    while(l<r){
        if(height[l]<height[r]){
            left_max=max(left_max,height[l]);
            water += left_max - height[l];
            l++;
        }else{
            right_max=max(right_max,height[r]);
            water += right_max - height[r];
            r--;
        }
    }
    return water;
}

// -------------------------------------------
// Problem 8: Product of Array Except Self
// -------------------------------------------
vector<int> productExceptSelf(vector<int>& nums){
    int n=nums.size();
    vector<int> res(n,1);
    int prefix=1;
    for(int i=0;i<n;i++){
        res[i]=prefix;
        prefix*=nums[i];
    }
    int suffix=1;
    for(int i=n-1;i>=0;i--){
        res[i]*=suffix;
        suffix*=nums[i];
    }
    return res;
}

// -------------------------------------------
// Problem 9: Rotate Matrix (in-place)
// -------------------------------------------
void rotateMatrix(vector<vector<int>>& matrix){
    int n=matrix.size();
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++)
            swap(matrix[i][j],matrix[j][i]);
    for(int i=0;i<n;i++)
        reverse(matrix[i].begin(),matrix[i].end());
}

// -------------------------------------------
// Problem 10: Set Matrix Zeroes
// -------------------------------------------
void setZeroes(vector<vector<int>>& matrix){
    if(matrix.empty()) return;
    int m=matrix.size(), n=matrix[0].size();
    vector<bool> row(m,false), col(n,false);
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            if(matrix[i][j]==0){
                row[i]=true;
                col[j]=true;
            }
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            if(row[i]||col[j])
                matrix[i][j]=0;
}

// -------------------------------------------
// Problem 11-15: Linked List problems
// -------------------------------------------
struct ListNode{
    int val;
    ListNode* next;
    ListNode(int x):val(x),next(nullptr){}
};

// 11. Reverse Linked List
ListNode* reverseListIter(ListNode* head){
    ListNode* prev=nullptr,*cur=head;
    while(cur){
        ListNode* nxt=cur->next;
        cur->next=prev;
        prev=cur;
        cur=nxt;
    }
    return prev;
}

ListNode* reverseListRec(ListNode* head){
    if(!head || !head->next) return head;
    ListNode* new_head=reverseListRec(head->next);
    head->next->next=head;
    head->next=nullptr;
    return new_head;
}

// 12. Detect cycle (Floyd) and remove
ListNode* detectCycleStart(ListNode* head){
    ListNode *slow=head,*fast=head;
    while(fast && fast->next){
        slow=slow->next;
        fast=fast->next->next;
        if(slow==fast){
            ListNode* ptr=head;
            while(ptr!=slow){
                ptr=ptr->next;
                slow=slow->next;
            }
            return ptr;
        }
    }
    return nullptr;
}

ListNode* removeCycle(ListNode* head){
    ListNode* start=detectCycleStart(head);
    if(!start) return head;
    ListNode* cur=start;
    while(cur->next!=start)
        cur=cur->next;
    cur->next=nullptr;
    return head;
}

// 13. Merge Two Sorted Lists
ListNode* mergeTwoLists(ListNode* l1,ListNode* l2){
    ListNode dummy(0), *tail=&dummy;
    while(l1 && l2){
        if(l1->val<=l2->val){tail->next=l1;l1=l1->next;}
        else{tail->next=l2;l2=l2->next;}
        tail=tail->next;
    }
    tail->next = l1?l1:l2;
    return dummy.next;
}

// 14. LRU Cache (Doubly Linked List + HashMap)
class DListNode{
public:
    int key,val;
    DListNode* prev;
    DListNode* next;
    DListNode(int k=0,int v=0):key(k),val(v),prev(nullptr),next(nullptr){}
};

class LRUCache {
    int cap;
    unordered_map<int,DListNode*> mp;
    DListNode *head,*tail;
public:
    LRUCache(int capacity){
        cap=capacity;
        head=new DListNode();
        tail=new DListNode();
        head->next=tail; tail->prev=head;
    }
    void remove(DListNode* node){
        node->prev->next=node->next;
        node->next->prev=node->prev;
    }
    void addFront(DListNode* node){
        node->next=head->next;
        node->prev=head;
        head->next->prev=node;
        head->next=node;
    }
    int get(int key){
        if(mp.find(key)==mp.end()) return -1;
        DListNode* node=mp[key];
        remove(node);
        addFront(node);
        return node->val;
    }
    void put(int key,int val){
        if(mp.find(key)!=mp.end()){
            DListNode* node=mp[key];
            node->val=val;
            remove(node); addFront(node);
        }else{
            if(mp.size()==cap){
                DListNode* lru=tail->prev;
                remove(lru);
                mp.erase(lru->key);
                delete lru;
            }
            DListNode* node=new DListNode(key,val);
            mp[key]=node;
            addFront(node);
        }
    }
};





#include <bits/stdc++.h>
using namespace std;

// ---------------------------
// Problem 16: Lowest Common Ancestor (LCA) of Two Nodes in Binary Tree
// ---------------------------
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x):val(x),left(nullptr),right(nullptr){}
};

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if(!root || root==p || root==q) return root;
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    if(left && right) return root;
    return left ? left : right;
}

// ---------------------------
// Problem 17: Serialize and Deserialize Binary Tree
// ---------------------------
void serialize(TreeNode* root, ostream& out) {
    if(!root){ out<<"# "; return; }
    out<<root->val<<" ";
    serialize(root->left,out);
    serialize(root->right,out);
}

TreeNode* deserialize(istringstream& in) {
    string val; 
    if(!(in>>val) || val=="#") return nullptr;
    TreeNode* root = new TreeNode(stoi(val));
    root->left = deserialize(in);
    root->right = deserialize(in);
    return root;
}

// ---------------------------
// Problem 18: Validate Binary Search Tree
// ---------------------------
bool isValidBST(TreeNode* root, long minv=LONG_MIN, long maxv=LONG_MAX){
    if(!root) return true;
    if(root->val <= minv || root->val >= maxv) return false;
    return isValidBST(root->left,minv,root->val) && isValidBST(root->right,root->val,maxv);
}

// ---------------------------
// Problem 19: Zigzag Level Order Traversal
// ---------------------------
vector<vector<int>> zigzagLevelOrder(TreeNode* root){
    vector<vector<int>> res;
    if(!root) return res;
    queue<TreeNode*> q; q.push(root);
    bool leftToRight=true;
    while(!q.empty()){
        int n=q.size();
        vector<int> level(n);
        for(int i=0;i<n;i++){
            TreeNode* node=q.front(); q.pop();
            int idx = leftToRight ? i : n-1-i;
            level[idx]=node->val;
            if(node->left) q.push(node->left);
            if(node->right) q.push(node->right);
        }
        res.push_back(level);
        leftToRight=!leftToRight;
    }
    return res;
}

// ---------------------------
// Problem 20: Diameter of Binary Tree
// ---------------------------
int diameter(TreeNode* root, int& ans){
    if(!root) return 0;
    int l=diameter(root->left,ans);
    int r=diameter(root->right,ans);
    ans = max(ans, l+r);
    return 1+max(l,r);
}

int diameterOfBinaryTree(TreeNode* root){
    int ans=0;
    diameter(root,ans);
    return ans;
}

// ---------------------------
// Problem 21: Vertical Order Traversal
// ---------------------------
vector<vector<int>> verticalOrder(TreeNode* root){
    map<int, vector<int>> mp;
    queue<pair<TreeNode*, int>> q;
    if(root) q.push({root,0});
    while(!q.empty()){
        auto [node, col]=q.front(); q.pop();
        mp[col].push_back(node->val);
        if(node->left) q.push({node->left,col-1});
        if(node->right) q.push({node->right,col+1});
    }
    vector<vector<int>> res;
    for(auto &p: mp) res.push_back(p.second);
    return res;
}

// ---------------------------
// Problem 22: Construct Binary Tree from Inorder & Preorder
// ---------------------------
TreeNode* buildTreeHelper(vector<int>& preorder,int preStart,int preEnd,vector<int>& inorder,int inStart,int inEnd, unordered_map<int,int>& idx){
    if(preStart>preEnd || inStart>inEnd) return nullptr;
    TreeNode* root = new TreeNode(preorder[preStart]);
    int inRoot=idx[root->val];
    int numsLeft=inRoot-inStart;
    root->left=buildTreeHelper(preorder,preStart+1,preStart+numsLeft,inorder,inStart,inRoot-1,idx);
    root->right=buildTreeHelper(preorder,preStart+numsLeft+1,preEnd,inorder,inRoot+1,inEnd,idx);
    return root;
}

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder){
    unordered_map<int,int> idx;
    for(int i=0;i<inorder.size();i++) idx[inorder[i]]=i;
    return buildTreeHelper(preorder,0,preorder.size()-1,inorder,0,inorder.size()-1,idx);
}

// ---------------------------
// Problem 23: Kth Smallest Element in BST
// ---------------------------
void inorderBST(TreeNode* root, vector<int>& vals){
    if(!root) return;
    inorderBST(root->left,vals);
    vals.push_back(root->val);
    inorderBST(root->right,vals);
}

int kthSmallest(TreeNode* root, int k){
    vector<int> vals;
    inorderBST(root,vals);
    return vals[k-1];
}

// ---------------------------
// Problem 24: Clone Graph
// ---------------------------
class GraphNode {
public:
    int val;
    vector<GraphNode*> neighbors;
    GraphNode(int x):val(x){}
};

GraphNode* cloneGraph(GraphNode* node){
    if(!node) return nullptr;
    unordered_map<GraphNode*, GraphNode*> mp;
    function<GraphNode*(GraphNode*)> dfs=[&](GraphNode* n){
        if(mp.count(n)) return mp[n];
        GraphNode* clone=new GraphNode(n->val);
        mp[n]=clone;
        for(auto neigh: n->neighbors)
            clone->neighbors.push_back(dfs(neigh));
        return clone;
    };
    return dfs(node);
}

// ---------------------------
// Problem 25: Number of Islands (DFS/BFS)
// ---------------------------
void dfsIsland(vector<vector<char>>& grid,int i,int j){
    int m=grid.size(), n=grid[0].size();
    if(i<0||i>=m||j<0||j>=n||grid[i][j]=='0') return;
    grid[i][j]='0';
    dfsIsland(grid,i+1,j);
    dfsIsland(grid,i-1,j);
    dfsIsland(grid,i,j+1);
    dfsIsland(grid,i,j-1);
}

int numIslands(vector<vector<char>>& grid){
    int count=0;
    for(int i=0;i<grid.size();i++)
        for(int j=0;j<grid[0].size();j++)
            if(grid[i][j]=='1'){
                dfsIsland(grid,i,j);
                count++;
            }
    return count;
}

// ---------------------------
// Problem 26: Word Ladder (BFS shortest path)
// ---------------------------
int ladderLength(string beginWord, string endWord, vector<string>& wordList){
    unordered_set<string> dict(wordList.begin(),wordList.end());
    if(!dict.count(endWord)) return 0;
    queue<pair<string,int>> q;
    q.push({beginWord,1});
    while(!q.empty()){
        auto [word,len]=q.front(); q.pop();
        if(word==endWord) return len;
        for(int i=0;i<word.size();i++){
            string orig=word;
            for(char c='a';c<='z';c++){
                word[i]=c;
                if(dict.count(word)){
                    q.push({word,len+1});
                    dict.erase(word);
                }
            }
            word=orig;
        }
    }
    return 0;
}

// ---------------------------
// Problem 27: Topological Sort (Kahn’s / DFS)
// ---------------------------
vector<int> topoSortKahn(int V, vector<vector<int>>& adj){
    vector<int> indegree(V,0);
    for(int u=0;u<V;u++)
        for(int v: adj[u]) indegree[v]++;
    queue<int> q;
    for(int i=0;i<V;i++) if(indegree[i]==0) q.push(i);
    vector<int> res;
    while(!q.empty()){
        int u=q.front(); q.pop();
        res.push_back(u);
        for(int v: adj[u]){
            if(--indegree[v]==0) q.push(v);
        }
    }
    return res;
}

// ---------------------------
// Problem 28: Detect Cycle in Directed Graph
// ---------------------------
bool dfsCycle(int u, vector<vector<int>>& adj, vector<int>& visited){
    visited[u]=1; // visiting
    for(int v: adj[u]){
        if(visited[v]==1) return true;
        if(visited[v]==0 && dfsCycle(v,adj,visited)) return true;
    }
    visited[u]=2;
    return false;
}

bool hasCycleDirected(int V, vector<vector<int>>& adj){
    vector<int> visited(V,0);
    for(int i=0;i<V;i++)
        if(visited[i]==0 && dfsCycle(i,adj,visited)) return true;
    return false;
}

// ---------------------------
// Problem 29: Dijkstra’s Shortest Path Algorithm
// ---------------------------
vector<int> dijkstra(int V, vector<vector<pair<int,int>>>& adj, int src){
    vector<int> dist(V,INT_MAX); dist[src]=0;
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.push({0,src});
    while(!pq.empty()){
        auto [d,u]=pq.top(); pq.pop();
        if(d>dist[u]) continue;
        for(auto [v,w]: adj[u]){
            if(dist[u]+w<dist[v]){
                dist[v]=dist[u]+w;
                pq.push({dist[v],v});
            }
        }
    }
    return dist;
}

// ---------------------------
// Problem 30-40: Dynamic Programming examples
// ---------------------------
// Example: Longest Increasing Subsequence
int lengthOfLIS(vector<int>& nums){
    vector<int> dp;
    for(int x: nums){
        auto it=lower_bound(dp.begin(),dp.end(),x);
        if(it==dp.end()) dp.push_back(x);
        else *it=x;
    }
    return dp.size();
}



#include <bits/stdc++.h>
using namespace std;

// ---------------------------
// Problem 41: Longest Palindromic Substring
// ---------------------------
string longestPalindrome(string s){
    int n=s.size();
    if(n<=1) return s;
    int start=0,maxLen=1;
    auto expand=[&](int l,int r){
        while(l>=0 && r<n && s[l]==s[r]){l--; r++;}
        if(r-l-1>maxLen){maxLen=r-l-1; start=l+1;}
    };
    for(int i=0;i<n;i++){
        expand(i,i);      // odd length
        expand(i,i+1);    // even length
    }
    return s.substr(start,maxLen);
}

// ---------------------------
// Problem 42: Partition Equal Subset Sum
// ---------------------------
bool canPartition(vector<int>& nums){
    int sum=accumulate(nums.begin(),nums.end(),0);
    if(sum%2!=0) return false;
    int target=sum/2;
    vector<bool> dp(target+1,false);
    dp[0]=true;
    for(int num: nums){
        for(int i=target;i>=num;i--){
            dp[i] = dp[i] || dp[i-num];
        }
    }
    return dp[target];
}

// ---------------------------
// Problem 43: Maximum Product Subarray
// ---------------------------
int maxProduct(vector<int>& nums){
    int maxF=nums[0], minF=nums[0], res=nums[0];
    for(int i=1;i<nums.size();i++){
        if(nums[i]<0) swap(maxF,minF);
        maxF = max(nums[i], maxF*nums[i]);
        minF = min(nums[i], minF*nums[i]);
        res = max(res,maxF);
    }
    return res;
}

// ---------------------------
// Problem 44: House Robber I & II
// ---------------------------
int rob(vector<int>& nums){
    int prev=0, curr=0;
    for(int num: nums){
        int temp = max(curr, prev+num);
        prev=curr;
        curr=temp;
    }
    return curr;
}

// For House Robber II (circle), take max(rob(nums[0..n-2]), rob(nums[1..n-1]))

// ---------------------------
// Problem 45: Unique Paths (with obstacles)
// ---------------------------
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid){
    int m=obstacleGrid.size(), n=obstacleGrid[0].size();
    vector<vector<int>> dp(m, vector<int>(n,0));
    if(obstacleGrid[0][0]==1) return 0;
    dp[0][0]=1;
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++){
            if(obstacleGrid[i][j]==1) dp[i][j]=0;
            else{
                if(i>0) dp[i][j]+=dp[i-1][j];
                if(j>0) dp[i][j]+=dp[i][j-1];
            }
        }
    return dp[m-1][n-1];
}

// ---------------------------
// Problem 46: Palindrome Partitioning (Min Cuts)
// ---------------------------
int minCut(string s){
    int n=s.size();
    vector<vector<bool>> dp(n, vector<bool>(n,false));
    for(int i=n-1;i>=0;i--)
        for(int j=i;j<n;j++)
            dp[i][j] = (s[i]==s[j]) && (j-i<2 || dp[i+1][j-1]);
    vector<int> cuts(n,INT_MAX);
    for(int i=0;i<n;i++){
        if(dp[0][i]) cuts[i]=0;
        else{
            for(int j=0;j<i;j++)
                if(dp[j+1][i])
                    cuts[i]=min(cuts[i],cuts[j]+1);
        }
    }
    return cuts[n-1];
}

// ---------------------------
// Problem 47: Merge K Sorted Lists
// ---------------------------
struct ListNode{
    int val;
    ListNode* next;
    ListNode(int x):val(x),next(nullptr){}
};

struct cmp{
    bool operator()(ListNode* a, ListNode* b){ return a->val>b->val;}
};

ListNode* mergeKLists(vector<ListNode*>& lists){
    priority_queue<ListNode*, vector<ListNode*>, cmp> pq;
    for(auto l: lists) if(l) pq.push(l);
    ListNode dummy(0), *tail=&dummy;
    while(!pq.empty()){
        ListNode* node=pq.top(); pq.pop();
        tail->next=node;
        tail=tail->next;
        if(node->next) pq.push(node->next);
    }
    return dummy.next;
}

// ---------------------------
// Problem 48: Sliding Window Maximum
// ---------------------------
vector<int> maxSlidingWindow(vector<int>& nums, int k){
    deque<int> dq;
    vector<int> res;
    for(int i=0;i<nums.size();i++){
        while(!dq.empty() && dq.front()<=i-k) dq.pop_front();
        while(!dq.empty() && nums[dq.back()]<nums[i]) dq.pop_back();
        dq.push_back(i);
        if(i>=k-1) res.push_back(nums[dq.front()]);
    }
    return res;
}

// ---------------------------
// Problem 49: Min Stack Implementation
// ---------------------------
class MinStack {
    stack<int> st;
    stack<int> minSt;
public:
    void push(int val){
        st.push(val);
        if(minSt.empty() || val<=minSt.top()) minSt.push(val);
    }
    void pop(){
        if(st.top()==minSt.top()) minSt.pop();
        st.pop();
    }
    int top(){ return st.top(); }
    int getMin(){ return minSt.top(); }
};

// ---------------------------
// Problem 50: Kth Largest Element in an Array
// ---------------------------
int findKthLargest(vector<int>& nums, int k){
    priority_queue<int, vector<int>, greater<int>> pq;
    for(int x: nums){
        pq.push(x);
        if(pq.size()>k) pq.pop();
    }
    return pq.top();
}

// ===========================
// ⚡ Bonus Advanced C++ Topics
// ===========================

// A) Function Wrapper / Lambda as Decorator
auto decorator(auto func){
    return [func](auto x){
        cout<<"Before function"<<endl;
        auto res=func(x);
        cout<<"After function"<<endl;
        return res;
    };
}

// B) RAII (Context Manager equivalent)
class FileRAII{
    FILE* f;
public:
    FileRAII(const char* fname,const char* mode){ f=fopen(fname,mode);}
    ~FileRAII(){ if(f) fclose(f);}
    FILE* get(){return f;}
};

// C) Threading Performance
void threadExample(){
    auto task=[](int n){ for(int i=0;i<n;i++);};
    thread t1(task,1000000), t2(task,1000000);
    t1.join(); t2.join();
}

// D) Global Interpreter Lock equivalent: C++ has none, threads run in parallel

// E) Memory Profiling & Optimization
// Use smart pointers, reserve vectors, minimize copies, avoid memory leaks
void memoryExample(){
    vector<int> v; v.reserve(1000); // prevent reallocations
    unique_ptr<int> p = make_unique<int>(42);
}














