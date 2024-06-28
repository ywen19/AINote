#include <bits/stdc++.h>
using namespace std;

typedef unsigned int u32;

const int N = 67108864;

// a: 给定的数组
// l: 每个桶的最小值
// r: 每个桶的最大值
u32 a[N+1];
u32 l[N+1], r[N+1];

// 用于生成数据的就行了

u32 nextInt(u32 x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

void initData(u32* a, int n, int k, u32 seed) {
    for (int i = 0; i < n; ++i) {
        seed = nextInt(seed);
        a[i] = seed >> (32 - k);
    }
}

// 这是求解答案的函数，你需要对全局变量中的 a 数组求解 maxGap 问题
// n, k：意义与题目描述相符
// 返回值：即为答案（maxGap）
u32 maxGap(int n, int k) {
    // initialization
    const int m = 1<<26;  // 桶的个数
    memset(l, -1, sizeof(int)*m);  // 将l中的所有位置赋值为-1
    memset(r, -1, sizeof(int)*m);

    const int _k = max(k-26, 0);  //参数辅助后续用位运算代替除法
    for (int i=0; i<n; ++i) {
        u32 bl = a[i] >> _k;  //等价于a[i]除以2的_k次幂 -> 求出a[i]所在的桶
        // 更新对应的桶的l和r
        if (l[bl] == -1) 
            l[bl] = r[bl] = a[i];
        else if (a[i] < l[bl])
            l[bl] = a[i];
        else if (a[i] > r[bl])
            r[bl] = a[i];
    }

    // 统计答案
    u32 last = a[0];
    u32 ans = 0;
    for (int i=0; i<m; ++i) 
        if (l[i] != -1) {
            if (last > l[i])
                last = l[i];
            if (l[i]-last > ans) 
                ans = l[i] - last;
            last = r[i];
        }
    return ans;
}



int main() {
    int n, k;
    u32 seed;

    scanf("%d%d%u", &n, &k, &seed);
    initData(a, n, k, seed);

    u32 ans = maxGap(n, k);

    printf("%u\n", ans);
    return 0;
}
