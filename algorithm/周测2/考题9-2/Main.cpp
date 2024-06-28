#include <bits/stdc++.h>
using namespace std;

// ref: https://blog.csdn.net/qq_46231566/article/details/123246852
// 取后3位其实就是该幂对1000取模(余数)
// ref: https://www.quora.com/How-do-I-find-the-last-3-digits-of-1234-5678
// ref: https://www.geeksforgeeks.org/find-last-digit-of-ab-for-large-numbers/
// ref: https://oi-wiki.org/math/binary-exponentiation/

const int MOD = 1000;


// 计算一组数据的答案，即 a^b 的后三位
// a: 如题目所述
// b: 如题目所述
// 返回值：a^b 的后三位
int solve(int a, int b) {
    int ans = 1;

    while (b) {
        if (b & 1) ans = ans * a % MOD;
        a= a * a  % MOD;
        b >>= 1;
    }
    return ans;
}

// 代码实现结束

int main() {
    int T;
    scanf("%d", &T);

    while (T--) {
        int a, b;
        scanf("%d%d", &a, &b);
        a %= MOD;
        printf("%d\n", solve(a, b));
    }
    
    return 0;
}
