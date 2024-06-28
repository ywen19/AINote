#include <bits/stdc++.h>
using namespace std;

int getAnswer(int n, int m, string *matrix) {
    int ans = 0; // 记录答案
    int *height = new int[m + 2](); // 存放高度的数组, 是一个长度空间为：m+2的数组, 左右增加两个高度为0的哨兵
    stack<int> myStack; // 单调栈，记录的是在height数组中的位置(索引)，栈顶元素所对应的高度是最高的。

    // 算法开始，进行每一行的遍历，k表示当前所枚举的是哪一行
    for (int k = 0; k < n; ++k) {
        // 1. 对当前行的每一列进行遍历，更新当前行和当前列的高度
        for (int i = 1; i<=m; ++i) {
            // 注意遍历的i是从[1,m], 更新的i要做-1操作，索引从0开始
            if(matrix[k][i-1] == '.') {
                height[i]++;  // 如果匹配到了红色，那么高度自增1
            }
            else {
                // 匹配到绿色，那么高度降为0
                height[i] = 0;
            }
        }
        // 对每行存放的索引栈进行重置操作
        while (!myStack.empty()) {
            myStack.pop();
        }
        // 初始化栈，压入0，表示对卡位点占位，因为height[0] = 0, 
        myStack.push(0);

        // 单调栈算法，循环到m+1, 因为height[m+1] = 0;
        for (int i = 1; i <= m + 1; ++i) {
            while(height[myStack.top()] > height[i]) {
                int nowHeight = height[myStack.top()];
                myStack.pop();
                ans = max(ans, (i - myStack.top() - 1) * nowHeight);
            }
            myStack.push(i);
        }
    }

    delete[] height;
    return ans;
}

int main() {
    ios::sync_with_stdio(false);  // 读入优化
    
    int n, m;
    cin >> n >> m;
    
    // 输入字符形矩阵 数组
    string *matrix = new string[n]();
    
    // 连续输入n行矩阵
    for (int i = 0; i < n; ++i)
        cin >> matrix[i];
    
    cout << getAnswer(n, m, matrix) << endl;
    
    delete[] matrix;
    return 0;
}
