#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1000005;

struct Line {
    int k, b, id;

    void read() {scanf("%d%d", &k, &b);}
    void println() {printf("%d %d %d\n", k, b, id);}
};

bool cmp(const Line& a, const Line& b) {return a.k==b.k ? a.b>b.b : a.k>b.k;}
bool cmp_by_id(const Line& a, const Line& b) {return a.id<b.id;}

double find_intersect_x(const Line& a, const Line& b) {
    return (double)(b.b - a.b)/(a.k - b.k);
}

// temp: 当前检测的直线入栈，方便后续比较交点
Line lines[MAXN], temp[MAXN];
int n;


int main() {
    scanf("%d", &n);
    for (int i=1; i<=n; i++) {
        lines[i].read();
        lines[i].id = i;
    }

    // 先k后b降序排列所有直线
    sort(lines+1, lines+n+1, cmp);

    int top_cnt = 1;
    temp[1] = lines[1];

    for (int i=2; i<=n; i++) {
        // 情况1：斜率相同，则lines中排在前面的直线一定在后面直线的上方; 不把当前直线加入栈中因为没有交点
        if (lines[i].k == lines[i-1].k) continue;
        // 情况2: 斜率不相同，则在此直线与所有其他直线相交的最大点(x最大)左右才可能满足题目所求
        // 递减top_cnt用来追溯当前线段在哪些temp栈内的线段上（直到不满足在上方的条件）
        // 注意边界情况即top不能小于等于1
        while (find_intersect_x(lines[i], temp[top_cnt]) >= find_intersect_x(temp[top_cnt], temp[top_cnt-1]) && top_cnt>1) top_cnt--;

        top_cnt++;
        temp[top_cnt] = lines[i];

    }

    sort(temp+1, temp+top_cnt+1, cmp_by_id);
    for (int i=1; i<=top_cnt; i++) printf("%d ", temp[i].id);

    return 0;
}