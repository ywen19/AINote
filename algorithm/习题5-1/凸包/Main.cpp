#include <bits/stdc++.h>
using namespace std;

// ================= 代码实现开始 =================

typedef long long ll;
const int N = 300005;

// 存储二维平面点
struct ip {
    int x, y, i;
    ip(int x = 0, int y = 0) : x(x), y(y), i(0) { }
    void ri(int _i) {
        scanf("%d%d", &x, &y);
        i = _i;
    }
};

// iv表示一个向量类型，其存储方式和ip一样
typedef ip iv;

// 先比较x轴再比较y轴，
bool operator < (const ip &a, const ip &b) {
    return a.x == b.x ? a.y < b.y : a.x < b.x;
}

// 两点相减得到的向量
iv operator - (const ip &a, const ip &b) {
    return iv(a.x - b.x, a.y - b.y);
}

// 计算a和b的叉积（外积）
ll operator ^ (const iv &a, const iv &b) {
    return (ll)a.x * b.y - (ll)a.y * b.x;
}



// 计算二维点数组a的凸包，将凸包放入b数组中，下标均从0开始
// a, b：如上
// n：表示a中元素个数
// 返回凸包元素个数
int convex(ip *a, ip *b, int n) {
    // 升序排序
    sort(a, a+n);

    // 若题目中有重复点，必须去重
    // n = unique(a, a+n) - a; // cannot use it here since we didn't override == to corresponding calculation

    int m = 0;
    // 求下凸壳
    for (int i=0; i<n; ++i) {
        for (; m>1 && ((b[m-1]-b[m-2])^(a[i]-b[m-2]))>0; --m);
        b[m++] = a[i];
    }

    // 求上凸壳
    for (int i = n-2, t=m; i>=0; --i) {
        for (; m>t && ((b[m-1]-b[m-2])^(a[i]-b[m-2]))>0; --m);
        b[m++] = a[i];
    }
    return m-1;

}

ip a[N], b[N];

int main() {
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; ++i)
        a[i].ri(i + 1);
    int m = convex(a, b, n), ans = m;
    for (int i = 0; i < m; ++i)
        ans = ((ll)ans * b[i].i) % (n + 1);
    printf("%d\n", ans);
    return 0;
}
