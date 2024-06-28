#include <bits/stdc++.h>
using namespace std;
// ================= 代码实现开始 =================
typedef pair<int,int> pii;
#define fi first
#define se second
const int N =2003;
//mind:mind[i][j]表示从初始状态到达状态(i,j)需要的最少步数
//q：用数组模拟的队列
//qh：队头下标
//qt：队尾下标
int mind[N][N];
pii q[N*N];
int qh,qt;
//倒水函数，表示状态p经过第k种策略倒水后的状态
//p：初始状态
//k：策略
//n，m：两杯子容量
//返回值：最终状态
pii to(pii p,int k,int n,int m){
    if(k == 0)//倒空杯子一
        return pii(0,p.se);
    else if(k == 1)//倒空杯子二
        return pii(p.fi,0);
    else if(k == 2)//倒满杯子一
        return pii(n,p.se);
    else if(k == 3)//倒满杯子二
        return pii(p.fi,m);
    else if(k == 4)//将杯子二的水向杯子一倒
        return pii(min(p.fi + p.se, n), max(p.fi + p.se -n, 0));
    else if(k == 5)//将杯子一的水向杯子二倒
        return pii(max(p.fi + p.se - m, 0), min(p.fi + p.se, m));
    else//啥也不干
        return p;
}
// 计算答案的函数
// n, m, t, d：意义均与题目描述一致
// 返回值：即为答案
int getAnswer(int n, int m, int t, int d) {
    //初始化，清空队列，将mind所有位置置为-1，表示未访问
    memset(mind,-1,sizeof(mind));
    qh = qt = 0;
    q[++qt] = pii(0,0);
    mind[0][0] = 0;
    //进行BFS
    while(qh < qt){
        pii u = q[++qh];//取出对头元素
        if(mind[u.fi][u.se] == t)
            break;//如果已经进行了t步，那么美必要继续搜索了，退出循环即可
        for(int k = 0; k < 6; ++k){//枚举6种策略
            pii v = to(u, k, n, m);
            if(mind[v.fi][v.se] != -1)//判断目标状态是否未曾到达过
                continue;
            q[++qt] = v;//加入队列
            mind[v.fi][v.se] = mind[u.fi][u.se] + 1;//记录mind
        }
    }
    //统计答案，找到最优的状态
    int ans = d;
    for(int i = 0; i <= n; ++i)
        for(int j=0; j <=m; ++j)
            if(mind[i][j] != -1)
                ans = min(ans,abs(i+j-d));
    return ans;
}

// ================= 代码实现结束 =================

int main() {
    int n, m, t, d;
    scanf("%d%d%d%d", &n, &m, &t, &d);
    int ans = getAnswer(n, m, t, d);
    printf("%d\n", ans);
    return 0;
}
