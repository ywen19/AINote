#include <bits/stdc++.h>
using namespace std;


int main() {
    int n;
    cin >> n;
    stack<int> myStack;
    for (int i = 0; i < n; ++i) {
        int tmp;
        cin >> tmp;
        myStack.push(tmp);
    }
    
    stack<int> help;

    while (!myStack.empty()) {
        int temp = myStack.top();
        myStack.pop();

        while (!help.empty() && temp<help.top()) {
            // 将help中比当前元素大的元素压入原始栈中
            myStack.push(help.top());
            help.pop();
        }
        help.push(temp);  //直到大小正确时压入help中
    }

    while (!help.empty()) {
        // 反顺序的元素压入栈中
        int temp = help.top();
        myStack.push(temp);
        help.pop();
    }

    while (!myStack.empty()) {
        cout << myStack.top() <<endl;
        myStack.pop();
    }
    return 0;
}
