#include <bits/stdc++.h>
using namespace std;

// ================= 代码实现开始 =================

const int N = 10005;

// 为了减少复制开销，我们直接读入信息到全局变量中，并统计了每个点的入度到数组in中
// n, m：点数和边数
// in：in[i]表示点i的入度
// e：e[i][j]表示点i的第j条边指向的点
int n, m, in[N];
vector<int> e[N];


// 判断所给有向无环图是否存在唯一的合法数列
// 返回值：若存在返回1；否则返回0。
bool getAnswer() {
    // 找到一个入度为0的点；有向无环图中至少存在一个入度为0的点
    // 若存在多个入度为0的点，说明合法数列不唯一
    int x = 0;
    for (int i=1; i<=n; ++i) {
        if (in[i] == 0) {
            if (x != 0) //之前已经找到一个
                return 0;
            x = i;
        }
    }

    // x表示的就是图中唯一的入度为0的点 
    for (int _=1; _<=n; ++_) {
        int z = 0;
        for (int i=0; i<(int)e[x].size(); ++i) {
            int y = e[x][i];
            --in[y];  //去除x-y这个边
            if (in[y] == 0) {
                if (z!= 0) 
                    return 0;
                z = y;
            }
        }
        x = z;
    }
    return 1;
}


int main() {
    int T;
    for (scanf("%d", &T); T--; ) {
        scanf("%d%d", &n, &m);
        for (int i = 1; i <= n; ++i) {
            in[i] = 0;
            e[i].clear();
        }
        for (int i = 0; i < m; ++i) {
            int x, y;
            scanf("%d%d", &x, &y);
            e[x].push_back(y);
            ++in[y];
        }
        printf("%d\n", getAnswer());
    }
    return 0;
}

