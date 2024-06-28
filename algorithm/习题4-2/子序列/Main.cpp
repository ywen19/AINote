#include <bits/stdc++.h>
using namespace std;

const int N = 500005, mo = 23333;

// f: f[i]表示前i个字符形成的不同的子序列个数
// p: p[i]表示字符s[i]最后出现的位置
// last: last[i]表示字符i最后出现的位置
int f[N], p[N], last[26];

// 为了减少复制开销，我们直接读入信息到全局变量中
// s：题目所给字符串，下标从1开始
// n：字符串长度
int n;
char s[N];

// 求出字符串s有多少不同的子序列
// 返回值：s不同子序列的数量，返回值对mo取模
int getAnswer() {
    // 计算p数组
    for (int i = 1; i<=n; ++i) {
        int a = s[i]-'a';
        p[i] = last[a];
        last[a] = i;
    }

    // 动态规划
    for (int i=1; i<=n; ++i) {
        if (p[i]==0) f[i] = f[i-1] + f[i-1] +1;
        else f[i] = f[i-1]+f[i-1]-f[p[i]-1];
        f[i] %= mo;
    }
    return (f[n]+mo)%mo;
}

// ================= 代码实现结束 =================

int main() {
    scanf("%s", s + 1);
    n = strlen(s + 1);
    printf("%d\n", getAnswer());
    return 0;
}
