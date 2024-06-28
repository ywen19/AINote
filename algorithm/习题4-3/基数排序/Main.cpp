#include <bits/stdc++.h>
using namespace std;

// ================= 代码实现开始 =================

typedef unsigned int u32;

// 以下代码不需要解释，你只需要知道这是用于生成数据的就行了

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

u32 hashArr(u32* a, int n) {
    u32 x = 998244353, ret = 0;
    for (int i = 0; i < n; ++i) {
        ret ^= (a[i] + x);
        x = nextInt(x);
    }
    return ret;
}


const int N = 100000000;
u32 a[N + 1];
u32 _a[N + 1];
const int m = 16;
const int B = 1 << m;
const int b = B - 1;
int sum[B + 1];


// 这是你的排序函数，你需要将全局变量中的 a 数组进行排序
// n, k：意义与题目描述相符
// 返回值：本函数需不要返回值（你只需要确保 a 被排序即可）
void sorting(int n, int k) {
   memset(sum, 0, sizeof(sum));
   for(int i = 0; i < n; ++i)
      ++sum[a[i] & b];
   for(int i = 1; i < B; ++i)
      sum[i] += sum[i - 1];
   for(int i = n - 1; i >= 0; --i)
      _a[--sum[a[i] & b]] = a[i];
   memset(sum, 0, sizeof(sum));
   for(int i = 0; i < n; ++i)
      ++sum[(_a[i] >> m) & b];
   for(int i =1; i < B; ++i)
      sum[i] += sum[i - 1];
   for(int i = n - 1; i >= 0; --i)
      a[--sum[(_a[i] >> m) & b]] = _a[i];
}

int main() {
    int n, k;
    u32 seed;
    scanf("%d%d%u", &n, &k, &seed);
    initData(a, n, k, seed);

    sorting(n, k);

    u32 ans = hashArr(a, n);
    printf("%u\n", ans);
    return 0;
}
