#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1005;
typedef pair<int, int> pii;

pii points[MAXN];  // store <x-coord, score> 

int dp[MAXN][MAXN];

int main() {
    int n;
    scanf("%d", &n);

    for (int i=1; i<=n; ++i) {
        int x, y;
        scanf("%d%d", &x, &y);
        points[i] = pii (x, y);
    }

    int ans = 0;
    // 需要考虑到向左或者向右两种方向
    // todo: 试了将两个方向合并但是逻辑上有问题, try again?
    for (int dir=0; dir<2; ++dir) {
        sort(points+1, points+n+1);  // sort by x coordinate

        for (int i=1; i<=n; i++) {
            dp[i][i] = points[i].second;
            for (int j=1; j<i; j++) {
                dp[i][j] = 0;
                for (int k = j; k && 2*points[j].first<=points[i].first+points[k].first; --k)
                    dp[i][j] = max(dp[i][j], dp[j][k]);
                ans = max(ans, dp[i][j]+=points[i].second);
            }
        }

        // make all x coord negative
        // so that after sort in the second round, we will use a reverse direction
        for (int i=1; i<=n; ++i) points[i].first *= -1;

    }

    printf("%d\n", ans);
    return 0;
}
