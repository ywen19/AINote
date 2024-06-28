#include <bits/stdc++.h>
using namespace std;

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

        bool unite(int x1, int x2) {
            int root_x1 = find(x1), root_x2 = find(x2);
            if (root_x1 != root_x2) {
                parent[root_x1] = root_x2;
                return true;
            }
            return false;
        }
};


vector<int> getAnswer(int n, int m, vector<int> U, vector<int> V) {
    dsuf union_find(n+1);
    vector<int> ans;

    // construct the maximum spanning tree
    for (int i=m-1; i>-1; --i) {
        if (union_find.unite(U[i], V[i])) ans.push_back(i+1);
    }
    reverse(ans.begin(), ans.end());
    return ans;
}


int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    vector<int> U, V;
    for (int i = 0; i < m; ++i) {
        int u, v;
        scanf("%d%d", &u, &v);
        U.push_back(u);
        V.push_back(v);
    }
    vector<int> ans = getAnswer(n, m, U, V);
    printf("%d\n", int(ans.size()));
    for (int i = 0; i < int(ans.size()); ++i)
        printf("%d\n", ans[i]);
    return 0;
}
