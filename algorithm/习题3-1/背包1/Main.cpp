#include <bits/stdc++.h>
using namespace std;

const int N = 5005;

int f[N];

// n：物品个数
// V：背包的体积
// t：长度为n的数组，第i个元素若为0，表示物品i为单个物品；若为1，表示物品i为多个物品。（i下标从0开始，下面同理）
// w：长度为n的数组，第i个元素表示第i个物品的价值
// v：长度为n的数组，第i个元素表示第i个物品的体积
// 返回值：最大价值之和
int getAnswer(int n, int V, vector<int> t, vector<int> w, vector<int> v) {
    for(int i=0; i<n; ++i)
        if(t[i] == 0)
             for(int j=V; j>=v[i]; --j)
                   f[j] = max(f[j],f[j-v[i]] + w[i]);
         else//完全背包，顺序枚举
             for(int j=v[i]; j<=V; ++j)
                   f[j] = max(f[j],f[j-v[i]]+w[i]);
    return f[V];
}

// ================= 代码实现结束 =================

int main() {
    int n, V;
    scanf("%d%d", &n, &V);
    vector<int> T, W, _V;
    for (int i = 0; i < n; ++i) {
        int t, w, v;
        scanf("%d%d%d", &t, &w, &v);
        T.push_back(t);
        W.push_back(w);
        _V.push_back(v);
    }
    printf("%d\n", getAnswer(n, V, T, W, _V));
    return 0;
}
