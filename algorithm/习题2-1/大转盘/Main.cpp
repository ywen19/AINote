#include <iostream>
#include <vector>
using namespace std ;

// allOne: 全1的二进制数，用于进行二进制“与”运算，allOne = 2^(n-1)-1
// vis: vis[i][u]表示从u出发值为i的边
// ans: 答案
int allOne;
vector<bool> vis[2];
string ans;

int twoPow(int x) {
    // calculate 2^x
    return 1<<x;
}

void dfs(int u) {
    // get the euler circuit
    for (int i=0; i<2; ++i) {
        if(!vis[i][u]) {
            int v = ((u<<1) | i) & allOne; //u左移1位，讲最低位置为i,再将最高位去掉
            vis[i][u] = 1;
            dfs(v);
            ans.push_back('0'+i);  //递归v、加入数字到ans中
        }
    }
}

// 本函数求解大转盘上的树，你需要把大转盘上的数按顺时针方向返回
// n: 对应转盘的大小
string getAnswer(int n) {
    allOne = twoPow(n-1) -1;
    ans = "";
    for (int i=0; i<2; ++i)
        vis[i].resize(twoPow(n-1), 0);
    
    dfs(0);
    return ans;
}

int main() {
    int n;
    scanf("%d", &n);
    cout << getAnswer(n) << endl;
    return 0;
}