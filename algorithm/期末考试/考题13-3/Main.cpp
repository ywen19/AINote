#include <bits/stdc++.h>
using namespace std;

constexpr int MAXN = 5000;

int values[MAXN + 10];

// O(n^2 * k) ref: https://leetcode.cn/problems/allocate-mailboxes/solutions/519023/an-pai-you-tong-by-leetcode-solution-t4oz/
// O(knlogn) ref: https://leetcode.cn/problems/allocate-mailboxes/solutions/287551/dong-tai-gui-hua-shi-jian-fu-za-du-oknlognkong-jia/

int pre_sum[MAXN+10];  // prefix sum from [1, i]

/*
O(n^2)的dp[k][n]表示将区间[1,n]分成k段的最优解,
这里用滚动数组的形式去简化了重复的计算（根据prefix sum储存的信息）；
且，位数是一个最优解。

观察这种形式的转移,记使得dp[j]=dp[p]+cost(p+1,j)的最小p为j的最优决策点p[j], 
可以证明: p[j]<=p[j+1]

也就是转移具有决策单调性
*/
double dp[MAXN+10], ndp[MAXN + 1];  // 记录差的和

int cost_to_median(int start, int end) {
    int center = (start + end) >> 1;

    return (pre_sum[end] - pre_sum[center] - (end-center)*values[center])  
        + ((center-start + 1)*values[center] - pre_sum[center] + pre_sum[start-1]);
}

void decision(int L, int R, int l, int r) {
    int mid = (L + R)/2, partition_pos = l;
    for(int i=l+1; i<=r && i<mid; i++) {
        if(dp[i]+cost_to_median(i+1, mid) < dp[partition_pos]+cost_to_median(partition_pos+1, mid)) {
            partition_pos = i;
        }
    }
    ndp[mid] = dp[partition_pos] + cost_to_median(partition_pos + 1, mid);
    if(L < mid) decision(L, mid-1, l, partition_pos);
    if(mid < R) decision(mid+1, R, partition_pos, r);
}


int main() {
    int n, k;
    scanf("%d%d", &n, &k);

    for (int i = 1; i <= n; ++i) scanf("%d", values + i);
    // 升序排序
    sort(values+1, values+n+1);

    //  前缀区间不分割（或者说分割成一个区间）
    pre_sum[0] = 0;
    for (int i=1; i<=n; ++i) {pre_sum[i] = pre_sum[i-1] + values[i];}

    // 在[1,i]中只取1个值（排序序列想cost最小则取中值）
    for(int i = 1; i <= n; i += 1) dp[i] = 1e15;

    // 递推分割数大于1小于等于k的情况
    for (int j=1; j<=k; j++) {
        decision(1, n, 0, n-1);
        for(int i = 1; i <= n; i += 1) dp[i] = ndp[i];
    }

    // 计算答案
    printf("%.4lf\n", (double)dp[n]);
    return 0;
}
