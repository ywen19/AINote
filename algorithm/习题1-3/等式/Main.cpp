#include <bits/stdc++.h>
using namespace std;

/*
inspired by: https://leetcode.cn/problems/satisfiability-of-equality-equations/solutions/279091/deng-shi-fang-cheng-de-ke-man-zu-xing-by-leetcode-/

reason not use rank optimization: https://blog.csdn.net/Qiuker_jl/article/details/109708771
path compression and rank can both decrease the query complexity to O(1);
path compression has higher degree of optimization(but will change the tree architecture -> not that important in this case).
*/

class dsuf {
    private:
        vector<int> parent;
    public:
        dsuf(int n) {
            parent.resize(n);
            iota(parent.begin(), parent.end(), 0);
        }

        int find(int x) {
            if (x == parent[x]) return x;
            
            parent[x] = find(parent[x]);
            return parent[x];
        }

        void unite(int x1, int x2) {
            parent[find(x1)] = find(x2);
        }
};


string getAnswer(int n, int m, vector<int> A, vector<int> B, vector<int> E) {
    dsuf union_find(n+1);

    // we will handle all union first, then detect if any inequality does not hold
    int cnt = 0;
     for(int i=0; i<m; ++i)
        if(E[i] == 1) {
           swap(E[i], E[cnt]);
           swap(A[i], A[cnt]);
           swap(B[i], B[cnt]);
           ++cnt;
     }

    for (int i=0; i<m; ++i) {
        if (E[i] == 1) {
           union_find.unite(A[i], B[i]);
        }
        if (E[i] == 0) {
            int setA_root = union_find.find(A[i]);
            int setB_root = union_find.find(B[i]);
            if (setA_root == setB_root) return "No";
        }
    }

    return "Yes";
}



int main() { 
    int T;
    for (scanf("%d", &T); T--; ) {
        int n, m;
        scanf("%d%d", &n, &m);
        vector<int> A, B, E;
        for (int i = 0; i < m; ++i) {
            int a, b, e;
            scanf("%d%d%d", &a, &b, &e);
            A.push_back(a);
            B.push_back(b);
            E.push_back(e);
        }
        printf("%s\n", getAnswer(n, m, A, B, E).c_str());
    }
    return 0;
}
