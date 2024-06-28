#include <bits/stdc++.h>
using namespace std;

// dp: 用于动态规划的数组，dp[i][j]表示走到第i行第j列能得到的最大数字和
vector<vector<int>> dp;

// n: 数字三角形大小
// a: 描述整个数字三角形,第i行的第j个数存在a[i][j]中，注意所有下标从1开始，即下标0储存的信息无效
int getAnswer(int n, vector<vector<int>> a) {
    dp.resize(n+1);
    for(int i=0; i<=n; ++i)
        dp[i].resize(i+2);
    for(int i = 1; i <= n; ++i)
        for(int j = 1; j<=i; ++j)
            dp[i][j] = max(dp[i-1][j-1],dp[i-1][j])+a[i][j];  //左上角和正上方的dp更新
    int ans = 0;
    for(int i=1; i<=n; ++i)
        ans = max(ans, dp[n][i]);  //最大数字之和
    return ans;
}

// ================= 代码实现结束 =================

int main() {
    int n;
    vector<vector<int>> a;
    scanf("%d", &n);
    a.resize(n + 1);
    for (int i = 0; i <= n; ++i)
        a[i].resize(i + 1);
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= i; ++j)
            scanf("%d", &a[i][j]);
    int ans = getAnswer(n, a);
    printf("%d\n", ans);
    return 0;
}
