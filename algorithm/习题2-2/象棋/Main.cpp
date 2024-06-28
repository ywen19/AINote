#include <bits/stdc++.h>
using namespace std;

const int N = 505*2, M= N * N;

struct E{
    // next: 下一条临接边, to: 本条边所指向的终点
    int next, to;
};

class Solution {
    int cnt;
    int ihead[N], mc[N];  // for a row and a column with the same start point in the board
    bool visited[N];
    E edge_info[M];

    public:
        void add_edge_info(int x, int y) {
            ++cnt;
            edge_info[cnt].next = ihead[x];
            edge_info[cnt].to = y;
            ihead[x] = cnt;
        }

        // Hungarian algorithm, 若找到增广路则返回true，否则返回false
        bool dfs_match(int x) {
            //x：x集上的点，从当前点出发找增广路
            //增广路（augmenting path）是始于非匹配点且终于非匹配点（除了起始的点）的交错路
            for (int i=ihead[x]; i!=0; i=edge_info[i].next) {
                int y = edge_info[i].to;
                //如果找到一个Y集上的点没有标记,标记此点
                if (!visited[y]) {
                    visited[y] = true;
                    //如果y是没有匹配点的，说明找到了一条增广路；或者说递归查找y的匹配点，得到了一条增广路
                    if (mc[y] == 0 || dfs_match(mc[y])) {
                        mc[x] = y;
                        mc[y] = x;
                        return true;
                    }
                }
            }
            return false;
        }

        int getAnswer(int n, vector<vector<int>> board) {
            cnt = 0;
            for(int i=1; i <= n * 2; ++i){
                    ihead[i] = 0;
                    mc[i] = 0;
            }

            //连边
            for(int i = 1; i <= n; ++i)
                for(int j = 1; j <= n; ++j)
                    if(board[i-1][j-1] == 1)
                        add_edge_info(i,j+n);
            
            int ans = 0;
            for(int i=1; i<=n; ++i)
                if(!mc[i]){
                    //如果x集中的第i个点没有匹配到Y集上的点，则从这个点出发寻找增广路
                    memset(visited, 0, sizeof(bool) * (n * 2));
                    if(dfs_match(i))
                        ++ans;//如果找到，答案直接+1
                }
            return ans;
        }
};



int main() {
    int n;
    scanf("%d", &n);
    vector<vector<int>> e;
    for (int i = 0; i < n; ++i) {
        vector<int> t;
        for (int j = 0; j < n; ++j) {
            int x;
            scanf("%d", &x);
            t.push_back(x);
        }
        e.push_back(t);
    }

    Solution solution;
    printf("%d\n", solution.getAnswer(n, e));

    return 0;
}