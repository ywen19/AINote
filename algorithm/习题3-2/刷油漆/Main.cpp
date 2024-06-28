#include <bits/stdc++.h>
using namespace std;

const int N = 21, mo = 23333;
//f:记忆已经计算过的答案，减少重复计算
int f[N][N][N][N][N][6];
//动态规划（记忆化搜索），求当车辆数目为a+b+c+d+e时涂油漆的方案数
//a:够涂1辆车的油漆种类数
//b:够涂2辆车的油漆种类数
//c:够涂3辆车的油漆种类数
//d:够涂4辆车的油漆种类数
//e:够涂5辆车的油漆种类数
//last:若last==2则表示前一辆车涂的油漆是从b中取出来的。last==3，从c中取出
//返回值：方案数
int dp(int a, int b, int c, int d, int e, int last){
    //n==0,返回1，即空也表示一种方案
    if((a|b|c|d|e) == 0)
        return 1;
    if(f[a][b][c][d][e][last] != -1)//如果之前算过答案，直接返回
        return f[a][b][c][d][e][last];
    long long ret = 0;
    //以下(last==2)等表达式的意思是：若这个表达式成立得到的是1否则是0
    if(a)
        //若last==2，则表示上一辆车是从b里取出来放到了a里，所以a中可以选择的油漆种类要少一个
        ret += dp(a-1, b, c, d, e, 1)*(a - (last==2));
    if(b)
        ret += dp(a+1, b-1, c, d, e, 2)*(b - (last==3));
    if(c)
        ret += dp(a, b+1, c-1, d, e, 3)*(c - (last==4));
    if(d)
        ret += dp(a, b, c+1, d-1, e, 4)*(d - (last==5));
    if(e)
        ret += dp(a, b, c, d+1, e-1, 5)*e;
    return f[a][b][c][d][e][last] = ret % mo;
}
int b[6];
// n辆车，m种油漆，第i种油漆够涂ai辆车，同时所有油漆恰好能涂完n辆车。若任意两辆相邻的车颜色不能相同，有多少种涂油漆的方案
// m：如题
// a：长度为m的数组，意义如题
// 返回值：方案数
int getAnswer(int m, vector<int> a) {
    memset(f, -1 ,sizeof f);
    for(int i=0; i<m; ++i)
        b[a[i]]++;
    return dp(b[1],b[2],b[3],b[4],b[5],0);
}

int main() {
    int m;
    scanf("%d", &m);
    vector<int> a;
    for (int i = 0; i < m; ++i) {
        int x;
        scanf("%d", &x);
        a.push_back(x);
    }
    printf("%d\n", getAnswer(m, a));
    return 0;
}
