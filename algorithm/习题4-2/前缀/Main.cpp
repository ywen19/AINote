#include <bits/stdc++.h>
using namespace std;

const int M = 505, L = 1000005;

// 基础Trie树问题

// c：trie树上的边，c[x][y]表示从节点x出发（x从1开始），字符为y的边（y范围是0到25）
// sz：sz[x]表示x节点的子树中终止节点的数量（子树包括x自身）
// cnt：trie树上节点的数目
int c[L][26], sz[L], cnt;

// 将字符串s加入到trie树中
// s：所要插入的字符串
void add(char *s) {
    int x = 0;
    for (; *s; ++s) {
        int y = *s -'a';  // 将字符范围变成0到25，分别对应字符a到z
        if (!c[x][y]) //若这个字符所对应的变不存在，新建一个节点
            c[x][y] = ++cnt;
        x = c[x][y]; //沿着边向下走
    }
    ++sz[x]; //x是终止节点
}

// 用于计算sz数组
// x：当前节点
void dfs(int x) {
    for (int y=0; y<26; ++y) {
        int z = c[x][y];
        if (z) {
            dfs(z);
            sz[x] += sz[z];
        }
    }
}

// 用字符串s沿着trie树上走，找到相应的节点
// s：所给字符串
// 返回值：走到的节点
int walk(char *s) {
    int x = 0;
    for (; *s; ++s) {
        int y = *s - 'a';
        if (!c[x][y]) //若边不存在直接返回不存在
            return 0;
        x = c[x][y];
    }
    return x;
}



char s[M];

int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    for (; n--;) {
        scanf("%s", s);
        add(s);
    }
    dfs(0);
    sz[0] = 0;
    for (; m--;) {
        scanf("%s", s);
        printf("%d\n", sz[walk(s)]);
    }
    return 0;
}
