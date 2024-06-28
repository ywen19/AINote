#include <bits/stdc++.h>
using namespace std;


vector<bool> isPrime, isDeng;


// 本函数求解质数或邓老师数（将这两个功能合并在了一起）
// n, k：意义均与题目描述相符
// 返回值：如果 k=0，则将所求的质数按从小到大的顺序放入返回值中；如果 k=1，则将所求的邓老师数按从小到大的顺序放入返回值中。
vector<int> getAnswer(int n, int k) {
    isPrime.resize(n+1, 1);
    isDeng.resize(n+1, 1);
    vector<int> ans;

    // starts from 2 to avoid cal value 1
    for (int i=2; i<=n; ++i) {
        if (isPrime[i]) isDeng[i]=0;
        if (k==0 && isPrime[i]) ans.push_back(i);
        if (k==1 && isDeng[i]) ans.push_back(i);

        for (int j=i+i; j<=n; j+=i) {
            isPrime[j] = 0;
            if (!isPrime[i]) isDeng[j]=0;
        }
    }
    return ans;
}


int main() {
    int n, k;
    scanf("%d%d", &n, &k);
    vector<int> ans = getAnswer(n, k);
    for (vector<int>::iterator it = ans.begin(); it != ans.end(); ++it)
        printf("%d\n", *it);
    return 0;
}
