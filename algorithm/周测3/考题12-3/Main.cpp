#include <bits/stdc++.h>

using namespace std;

using LL = long long;

constexpr int MAXN = 300;
constexpr int MAXA = 50000;

int a[MAXN + 10];
bool rep[MAXA + 10];

int main() {
    int n;
    scanf("%d", &n);
    for (int i=0; i<n; i++) scanf("%d", a+i);

    sort(a, a+n);
    n = unique(a, a+n) - a;

    int ans = 0;
    rep[0] = 1;
    for (int i=0; i<n; i++) {
        if (rep[a[i]]) continue;
        ans++;
        for (int j=a[i]; j<=a[n-1]; j++)
            if (rep[j-a[i]]) rep[j] = 1;
    }
    printf("%d\n", ans);
    return 0;
}
