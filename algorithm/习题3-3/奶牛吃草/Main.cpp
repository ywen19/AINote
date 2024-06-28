#include <bits/stdc++.h>
using namespace std;

const int N = 2000;
// dp[l][r][j]表示吃完[l, r]范围内的青草时的最小答案,j只有0/1两种取值,表示奶牛吃完最后一棵草停在青草l或r上
int dp[N+2][N+2][2];


// 本函数求解答案（损失的最小口感和）
// n：青草棵数
// k：奶牛的初始坐标
// x：描述序列 x（这里需要注意的是，由于 x 的下标从 1 开始，因此 x[0] 的值为 -1，你可以忽略它的值，只需知道我们从下标 1 开始存放有效信息即可），意义同题目描述
// 返回值：损失的最小口感和
int getAnswer(int n, int k, vector<int> x) {
    sort(x.begin()+1, x.end());
    for (int i=1; i<=n; ++i)
        dp[i][i][0] = dp[i][i][1] = abs(x[i]-k)*n;  //边界设置：只吃一棵草的情况下答案是？
    for (int len=1; len<n; ++len)
        for (int l=1, r; (r=l+len)<=n; ++l) {
            // 枚举区间（注意： 先枚举区间长度，再枚举左端点，并自然求出右端点
            // 进行转移
            dp[l][r][0] = min(dp[l+1][r][0]+(n-r+l)*abs(x[l]-x[l+1]), dp[l+1][r][1]+(n-r+l)*abs(x[l]-x[r]));
            dp[l][r][1] = min(dp[l][r-1][1]+(n-r+l)*abs(x[r]-x[r-1]), dp[l][r-1][0]+(n-r+l)*abs(x[r]-x[l]));
        }
    return min(dp[1][n][0], dp[1][n][1]);
}

// ================= 代码实现结束 =================

int main() {
    int n, k, tmp;
    vector<int> x;
    scanf("%d%d", &n, &k);
    x.clear();
    x.push_back(-1);
    for (int i = 1; i <= n; ++i) {
        scanf("%d", &tmp);
        x.push_back(tmp);
    }
    int ans = getAnswer(n, k, x);
    printf("%d\n", ans);
    return 0;
}
