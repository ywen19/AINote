#include <bits/stdc++.h>
using namespace std;

int n, m, s, t;
vector<int> edges[100005];
int weights[100005];

bool hasSolution(int val) {
    bool visited[100005];
    for (int i=1; i<=n; i++) visited[i]=0;

    queue<int> candidates;
    candidates.push(s);
    visited[s] = 1;

    while(!candidates.empty()) {
        int current = candidates.front();
        candidates.pop();
        for (int connected: edges[current]) {
            // if the weight is smaller than the given higher bound and not visited
            // else there is no solution for the current condition
            if (weights[connected]<=val && !visited[connected]) {
                // if already at terminal we could terminate
                if (connected==t) return 1;
                // else we see the current vertex as another candidate and see if 
                // there is a path under given condition
                candidates.push(connected);
                visited[connected] = 1;
            }
        }
    }
    return 0;  // no solution for the given condition
}


int main()
{
    scanf("%d%d", &n, &m);
    for (int i=1; i<=n; i++) scanf("%d", weights + i);
    while (m--) {
        int u, v;
        scanf("%d%d", &u, &v);
        edges[u].push_back(v);
        edges[v].push_back(u);
    }
    scanf("%d%d", &s, &t);

    // 用类似于二分法的方法，不知道这样递减对不对
    int lower = max(weights[s], weights[t]);  //答案必不可能小于两点的最大权值，除非没有通路则为-1
    // 找到最大的权值
    int higher = weights[1];
    for (int i=2; i<=n; i++) higher=max(higher, weights[i]);

    // 如果在什么都允许的情况下已经不存在解了，那么就不存在
    if (!hasSolution(higher)) {
        puts("-1");
        return 0;
    }

    int ans = higher--;
    while(lower<=higher) {
        int mid = (lower+higher)>>1;
        if (hasSolution(mid)) {higher=mid-1; ans=mid;}
        else {lower=mid+1;}
    }
    printf("%d\n", ans);
    return 0;
}