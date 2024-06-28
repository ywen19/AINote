#include <bits/stdc++.h>
using namespace std;
// ================= 代码实现开始 =================
typedef double lf;
typedef long long ll;
const int N = 300005;
// 用于存储一个二维平面上的点
struct ip {
    int x, y;
    // 构造函数
    ip(int x = 0, int y = 0) : x(x), y(y) { }
     // 先比较x轴，再比较y轴
    bool operator < (const ip &a) const {
        return x == a.x ? y < a.y : x < a.x;
    }
} a[N], b[N];
// 计算x的平方
ll sqr(const ll &x) {
    return x * x;
}
// 计算点a和点b的距离
lf dis(const ip &a, const ip &b) {
    return sqrt(sqr(a.x - b.x) + sqr(a.y - b.y));
}
lf ans;
//分治求最近点对
//l,r：表示闭区间[l,r]
void solve(int l, int r){
    //边界情况
    if(r - l <= 1){
        if(a[l].y > a[r].y)
            swap(a[l], a[r]);
        if(l != r)
            ans = min(ans,dis(a[l], a[r]));
        return;
    }
    //分治计算两遍
    int mid = (l + r) >> 1;
    int md = a[mid].x;//中间值
    solve(l, mid);
    solve(mid + 1, r);
    //对y轴进行归并排序
    int cnt = 0;
    for(int i = l, j = mid + 1; i <= mid || j <= r;){
        for(; i <= mid && md - a[i].x >= ans; ++i);
        for(; j <= r && a[j].x - md >= ans; ++j);
        if(i <= mid && (j > r || a[i].y < a[j].y))
            b[cnt++] = a[i++];
        else
            b[cnt++] = a[j++];
    }
    //现在b数组
    for(int i = 0; i < cnt; ++i)
        for(int j = i + 1; j < cnt && b[j].y - b[i].y < ans; ++j)
            ans = min(ans, dis(b[i], b[j]));
    cnt = 0;
    for(int i = l, j = mid + 1; i <= mid || j <=r;){
        if(i <= mid &&(j > r || a[i].y < a[j].y))
            b[cnt++] = a[i++];
        else
            b[cnt++] = a[j++];
    }
    memcpy(a + l, b, sizeof(ip) * cnt);
}
// 计算最近点对的距离
// n：n个点
// X, Y：分别表示x轴坐标和y轴坐标，下标从0开始
// 返回值：最近的距离
double getAnswer(int n, vector<int> X, vector<int> Y) {
   for(int i = 0; i < n; ++i)
        a[i+1] = ip(X[i], Y[i]);
   ans = 1e100;
   sort(a + 1,a + 1 + n);
   solve(1, n);
   return ans;
}


int main() {
    int n;
    scanf("%d", &n);
    vector<int> X, Y;
    for (int i = 1; i <= n; ++i) {
        int x, y;
        scanf("%d%d", &x, &y);
        X.push_back(x);
        Y.push_back(y);
    }
    printf("%.2f\n", getAnswer(n, X, Y));
    return 0;
}

