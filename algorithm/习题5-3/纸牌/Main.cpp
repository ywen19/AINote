#include <bits/stdc++.h>
using namespace std;

const int MAXN = 50005;

int self[MAXN], enemy[MAXN];
bool taken[MAXN * 2];  // 记录哪些牌已经被分给己方了


int main() {
    int n;
    scanf("%d", &n);

    for (int i=1; i<=n; i++) {
        scanf("%d", self+i);
        taken[self[i]] = 1;
    }

    // get the cards for the enemy
    int idx = 0;
    for (int i=1; i<=2*n; i++) {
        if (!taken[i]) enemy[++idx] = i;
    } // enemy cards should be sorted in ascending order by this loop

    sort(self+1, self+n+1);  // sorted self cards in ascending order by this loop

    int ans = 0, head = 1;
    // 类似于田忌赛马，一旦ans+1,我们将己方的拍的指针+1，意味着有一张拍已经出掉了
    for (int i=1; i<=n; i++) {
        if (self[head] < enemy[i]) {
            ++ans;
            ++head;
        }
    }
    printf("%d\n", ans);
    return 0;
}
