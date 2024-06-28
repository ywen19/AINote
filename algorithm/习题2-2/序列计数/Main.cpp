#include <bits/stdc++.h>
using namespace std;

const int N = 300005;

int n, d, max_value[N], min_value[N];
vector<int> a;


long long solve(int low_bound, int high_bound) {
    // bounds are the left and right boundary of a range
    if (low_bound == high_bound) return 0;

    int mid = (low_bound + high_bound) >> 1; // mid point
    long long ans = solve(low_bound, mid) + solve(mid+1, high_bound);

    // calculate the min and max values in [mid+1, high_bound]
    for (int i = mid+1; i<=high_bound; ++i) {
        min_value[i] = (i==mid+1) ? a[i] : min(min_value[i-1], a[i]);
        max_value[i] = (i==mid+1) ? a[i] : max(max_value[i-1], a[i]);
    }

    /* 倒序枚举子序列的左端点i, i在[low_bound, mid]
       pos表示若连续子序列的左端点是i，那么子序列的右端点最远能扩展到pos位置, pos取值范围[mid+1, high bound], 初始化为high_bound
       mn是后缀最小值, mx是后缀最大值, 也就是说mn = min(a[i..mid]), mx同理
       那么以i为左端点的子序个数应该有pos - mid个
    */
   int mn = 0, mx = 0, pos = high_bound;

   for (int i = mid; i>=low_bound && pos>mid; --i) {
        // update mn and max
        mn = (i==mid) ? a[i] : min(mn, a[i]);
        mx = (i==mid) ? a[i] : max(mx, a[i]);

        for (; pos>mid && max(mx, max_value[pos])-min(mn, min_value[pos])>d; --pos); 
        ans += pos-mid;
   }
   return ans;

}


// 求出有多少个a数组中的连续子序列（长度大于1），满足该子序列的最大值最小值之差不大于d
// n：a数组的长度
// d：所给d
// a：数组a，长度为n
// 返回值：满足条件的连续子序列的个数
long long getAnswer(int n, int d, vector<int> a) {
    ::n = n;
    ::d = d;
    ::a = a;
    return solve(0, n-1);
}

// ================= 代码实现结束 =================


int main() {
    int n, d;
    scanf("%d%d", &n, &d);
    vector<int> a;
    a.resize(n);
    for (int i = 0; i < n; ++i)
        scanf("%d", &a[i]);
    printf("%lld\n", getAnswer(n, d, a));
    return 0;
}
