#include <bits/stdc++.h>
using namespace std;

const int N = 100005;
typedef pair<int, int> pii;

/*
graph: graph[i] 是节点i的出边,其中first存储到达的节点, second存储权重
pq: 辅助dijkstra的优先队列
flag: 记录节点是否进过松弛
min_d：存储起点s到每个节点的最短路径
*/
vector<pii> graph[N];
priority_queue<pii, vector<pii>, greater<pii>> pq;
bool flag[N];
int min_d[N];

/*
本题是 Dijkstra 算法的模板练习题。]

[使用朴素的 Dijkstra 算法可以通过前 10 个测试点。]

[使用堆或__std::priority_queue__优化的 Dijkstra 算法可以通过所有测试点。
*/

// ================= 代码实现开始 =================

/* 请在这里定义你需要的全局变量 */

// 这个函数用于计算答案（最短路）
// n：节点数目
// m：双向边数目
// U,V,W：分别存放各边的两端点、边权
// s,t：分别表示起点、重点
// 返回值：答案（即从 s 到 t 的最短路径长度）
int shortestPath(int n, int m, vector<int> U, vector<int> V, vector<int> W, int s, int t) {
    // clear pq, graph, flag, min_d
    while(!pq.empty()) pq.pop();
    for (int i=0; i<n; ++i) graph[i].clear();
    memset(flag, 0, sizeof(flag));
    memset(min_d, 127, sizeof(min_d));

    // construct the graph
    for (int i=0; i<m; ++i) {
        graph[U[i]].push_back(make_pair(V[i], W[i]));
        graph[V[i]].push_back(make_pair(U[i], W[i]));
    }

    // 设置起点的最短路为0，并将此点加入优先队列
    min_d[s] = 0;
    pq.push(make_pair(0, s));

    // dijkstra
    while (!flag[t]) {
        // get the top element in the priority queue
        int u = pq.top().second;
        pq.pop();

        if (!flag[u]) {
            // 每个节点至多松弛（就是visit）一次
            flag[u] = 1;
            for (vector<pii>::iterator itr = graph[u].begin(); itr != graph[u].end(); ++itr) {
                // 枚举所有u出发的边
                int v = itr->first, w = itr->second;
                if (min_d[v] <= min_d[u]+w) continue;
                min_d[v] = min_d[u] + w; // answer update
                pq.push(make_pair(min_d[v], v));
            }
        }
    }
    return min_d[t];
}

// ================= 代码实现结束 =================

int main() {
    int n, m, s, t;
    scanf("%d%d%d%d", &n, &m, &s, &t);
    vector<int> U, V, W;
    U.clear();
    V.clear();
    W.clear();
    for (int i = 0; i < m; ++i) {
        int u, v, w;
        scanf("%d%d%d", &u, &v, &w);
        U.push_back(u);
        V.push_back(v);
        W.push_back(w);
    }

    printf("%d\n", shortestPath(n, m, U, V, W, s, t));
    return 0;
}
 
