#include <bits/stdc++.h>
using namespace std;

// n：意义如题
// height：高度数组，height[i]表示第i列的高度（下标从1开始），数组大小为n+2，其中height[0]和height[n+1]都为0
// 返回值：题目所求答案，即最大面积
int getAnswer(int n, int *h) {
    // 最后答案
    int ans = 0;
    stack<int> myStack;
    myStack.push(0);

    for (int i = 1; i <= n + 1; ++i) {
        // 如果栈顶元素高度大于当前元素的高度时，则弹出栈顶
        while(h[myStack.top()] > h[i]) {
            int nowHeight = h[myStack.top()];
            myStack.pop();
            ans = max(ans, (i - myStack.top() - 1) * nowHeight);
        }
        // 将当前下标插入栈中
        myStack.push(i);
    }
    return ans;
}

int main() {
    int n;
    cin >> n;
    
    int* height = new int[n + 2]();
    
    for (int i = 1; i <= n; ++i)
        cin >> height[i];
    
    cout << getAnswer(n, height) << endl;
    
    delete[] height;
    return 0;
}
