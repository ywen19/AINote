#include <bits/stdc++.h>
using namespace std;

// Next: KPM算法中的Next数组
vector<int> Next;

// 这是匹配函数，将所有匹配位置求出并返回
// n：串 A 的长度
// A：题目描述中的串 A
// m：串 B 的长度
// B：题目描述中的串 B
// 返回值：一个 vector<int>，从小到大依次存放各匹配位置
vector<int> match(int n, string A, int m, string B) {
   Next.resize(m);
   int j = Next[0]= -1;
   for(int i =1; i<m; ++i){
        while( j >= 0 &&B[i] != B[j+1])
            j = Next[j];
        if(B[i] == B[j+1])
            ++j;
        Next[i] = j;
   }
   j = -1;
   vector<int> ans;
   for(int i=0; i<n; ++i){
        while(j >=0 &&A[i] != B[j+1])
            j = Next[j];
        if(A[i] == B[j+1])
            ++j;
        if(j == m-1)
            ans.push_back(i-m+1);
   }
   return ans;
}


int main() {
    ios::sync_with_stdio(false);
    int n, m;
    string A, B;
    cin >> n >> A;
    cin >> m >> B;
    vector<int> ans = match(n, A, m, B);
    for (vector<int>::iterator it = ans.begin(); it != ans.end(); ++it)
        cout << *it << '\n';
    return 0;
}
