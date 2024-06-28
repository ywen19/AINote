#include <algorithm>
#include <iostream>
#include <cstring>
using namespace std;

constexpr int NUM = 5001;
constexpr int MAXDATA = 1e8 + 9;
int f[MAXDATA];

struct data_t {
    int t, e, h;
};

// 自定义比较函数，按照结构体内的 t 属性进行升序排序
bool cmp(data_t a, data_t b) {
    return a.t < b.t;
}

int main() {
    int h, n, s;
    data_t x[NUM]; // 添加 x 数组定义
    scanf("%d%d", &h, &n);
    scanf("%d", &s);

    // 读取数据
    for (int i = 1; i <= n; i++)
        scanf("%d%d%d", &x[i].t, &x[i].e, &x[i].h);

    // 按照出现时间进行排序
    sort(x + 1, x + 1 + n, cmp);

    int ans = -1;
    memset(f, -1, MAXDATA);
    f[0] = s;

    // 动态规划求解答案
    for (int i = 1; i <= n; i++) {
        for (int j = h; j >= 0; j--) {
            if (f[j] >= x[i].t) {
                if (j + x[i].h >= h) {
                    printf("%d\n", x[i].t);
                    return 0;
                }
                f[j + x[i].h] = max(f[j + x[i].h], f[j]);
                f[j] += x[i].e;
            }
        }
    }

    printf("-1\n");
    printf("%lld\n", f[0]);
    return 0;
}