#include <bits/stdc++.h>
using namespace std;

typedef pair<int, int> pii;

constexpr int MAXN = 100000;

// n: vertex amount
// m: edge amount
int n, m;
// graph: idx->start point; graph[idx] = [<end point, weight>, ...]
vector<pii> graph[MAXN+5];
// color: 染色法判定是否为二分图
int color[MAXN]; 

bool dfs(int i, int c, int w_thres) {
    color[i] = c;

    // 循环邻接点
    for (auto edge: graph[i]) {
        int neighbor = edge.first, w = edge.second;

        if (w <= w_thres) continue;

        if (color[neighbor]!=-1) {
            if (color[neighbor] == color[i]) return false;
        }
        else if (!dfs(neighbor, 1-c, w_thres)) return false;

        /*if (w > w_thres) {
            if (color[neighbor] == -1) dfs(neighbor, 1-c, w_thres);
            else if (color[neighbor] == color[i]) is_bipartite=false;
        }*/
    }
    return true;
}


bool hasSolution(int w_thres) {
    // -1: 未染色； 有0/1两种颜色
    memset(color, -1, sizeof(color));

    for (int i=0; i<=n; ++i) {
        if (color[i] == -1) {
            if (!dfs(i, 0, w_thres)) return false;
        }
    }
    return true;
}




int main() {
    scanf("%d%d", &n, &m);

    int l = 0, r = 0;  // 二分查找最小权值
    // 输入无向图信息
    int u, v, w;
    for (int i=0; i<m; ++i) {
        scanf("%d%d%d", &u, &v, &w);
        graph[u].emplace_back(make_pair(v, w));
        graph[v].emplace_back(make_pair(u, w));
        r = max(r, w);
    }

    // 开始二分
    int ans = -1;
    while (l <= r) {
        int mid = (l+r)>>1;

        // 如果小于mid时存在解， 更新搜索范围至[l, mid-1]
        // 否则，更新搜索范围至[mid+1, r];
        if (hasSolution(mid)) {
            r = mid - 1;
            ans = mid;
        }
        else l = mid + 1;
    }

    
    //计算答案至 ans;
    printf("%d\n", ans);

    return 0;
}
