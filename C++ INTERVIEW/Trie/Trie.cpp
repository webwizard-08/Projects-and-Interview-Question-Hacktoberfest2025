// Trie.cpp
// Simple Trie (prefix tree) implementation for lowercase a-z strings.
// Provides insert, search, startsWith, erase and a small demo in main.

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    bool isWord = false;
    array<TrieNode*, 26> next{};
    TrieNode() { next.fill(nullptr); }
    ~TrieNode() {
        for (auto p : next) if (p) delete p;
    }
};

class Trie {
public:
    Trie() : root(new TrieNode()) {}
    ~Trie() { delete root; }

    // Insert a lowercase word
    void insert(const string& word) {
        TrieNode* cur = root;
        for (char ch : word) {
            if (!valid(ch)) continue;
            int idx = ch - 'a';
            if (!cur->next[idx]) cur->next[idx] = new TrieNode();
            cur = cur->next[idx];
        }
        cur->isWord = true;
    }

    // Search exact word
    bool search(const string& word) const {
        TrieNode* node = findNode(word);
        return node && node->isWord;
    }

    // Check if any word starts with prefix
    bool startsWith(const string& prefix) const {
        return findNode(prefix) != nullptr;
    }

    // Erase a word. Returns true if erased (word existed).
    bool erase(const string& word) {
        return eraseRec(root, word, 0);
    }

private:
    TrieNode* root;

    static bool valid(char ch) {
        return ch >= 'a' && ch <= 'z';
    }

    TrieNode* findNode(const string& s) const {
        TrieNode* cur = root;
        for (char ch : s) {
            if (!valid(ch)) return nullptr;
            int idx = ch - 'a';
            if (!cur->next[idx]) return nullptr;
            cur = cur->next[idx];
        }
        return cur;
    }

    // Recursively erase word. Return true if parent should delete this node.
    bool eraseRec(TrieNode* node, const string& word, size_t depth) {
        if (!node) return false;
        if (depth == word.size()) {
            if (!node->isWord) return false; // word not present
            node->isWord = false;
            // if node has no children, signal to delete it
            return all_of(node->next.begin(), node->next.end(), [](TrieNode* p){ return p == nullptr; });
        }
        char ch = word[depth];
        if (!valid(ch)) return false;
        int idx = ch - 'a';
        TrieNode* child = node->next[idx];
        if (!child) return false;
        bool shouldDeleteChild = eraseRec(child, word, depth + 1);
        if (shouldDeleteChild) {
            delete child;
            node->next[idx] = nullptr;
            // if current node is not a word and has no children, tell parent to delete
            return !node->isWord && all_of(node->next.begin(), node->next.end(), [](TrieNode* p){ return p == nullptr; });
        }
        return false;
    }
};

// Small demo when run directly.
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Trie trie;
    vector<string> words = {"apple", "app", "apt", "banana", "band", "bandana"};
    for (auto &w : words) trie.insert(w);

    cout << boolalpha;
    cout << "search(\"app\"): " << trie.search("app") << '\n';
    cout << "search(\"apple\"): " << trie.search("apple") << '\n';
    cout << "search(\"ap\"): " << trie.search("ap") << '\n';
    cout << "startsWith(\"ap\"): " << trie.startsWith("ap") << '\n';
    cout << "startsWith(\"ban\"): " << trie.startsWith("ban") << '\n';

    cout << "erase(\"app\"): " << trie.erase("app") << '\n';
    cout << "search(\"app\"): " << trie.search("app") << '\n';
    cout << "search(\"apple\"): " << trie.search("apple") << '\n';

    cout << "erase(\"banana\"): " << trie.erase("banana") << '\n';
    cout << "search(\"banana\"): " << trie.search("banana") << '\n';

    return 0;
}