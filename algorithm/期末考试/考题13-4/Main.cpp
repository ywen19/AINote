#include <bits/stdc++.h>

using namespace std;

constexpr int MAXL = 2000000;


int main() {
    int k, ans=0;
    scanf("%d", &k);

    string p, t;
    cin >> p >> t;

    //cout<<p<<endl;
    //cout<<t<<endl;

    int p_len = p.length(), t_len = t.length();
    for (int i=0; i<=t_len-p_len; ++i) {
        int cnt_mismatch = 0;
        for (int j=0; j<p_len; ++j) {
            if (t[i+j] != p[j]) {
                ++cnt_mismatch;
                if (cnt_mismatch > k) break;
            }
        }
        if (cnt_mismatch<=k) ++ans;
    }

    printf("%d\n", ans);
    return 0;
}
