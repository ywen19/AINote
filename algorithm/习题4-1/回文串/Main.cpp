#include <bits/stdc++.h>
using namespace std;

const int N = 500005;
char s[N*2];  // 变化后的s数组
int len[N*2];  //每个位置能向左扩展的最大长度


// 计算str中有多少个回文子串
// 返回值：子串的数目
long long getAnswer(string str) {
    int n = str.size();

    // 将字符串变成#a#b#c这样的形式，#被认为是0
    for (int i=n; i; --i) {
        s[i<<1] = str[i-1], s[i<<1|1]=0;
    }

    // 边界(位置0和n+1)设为不同于#的东西（即1和2）
    n = n<<1|1;
    s[1]=0, s[0]=1, s[n+1]=2;

    // manacher算法
    int cur=1;
    long long ans = 0;
    for (int i=2; i<=n; ++i) {
        int &now = len[i], pos=(cur<<1)-i;
        now = max(min(len[pos], cur+len[cur]-i), 0);
        while(s[i-now-1]==s[i+now+1]) ++now;

        if(i+now>cur+len[cur]) cur=i;
        ans += (now+1)/2;
    }
    return ans;
}


char _s[N];

int main() {
    scanf("%s", _s + 1);
    printf("%lld\n", getAnswer(_s + 1));
    return 0;
}
