#include <bits/stdc++.h>
using namespace std;


// global variable
string myStack[10000];  // by instruction, maximum size is 10000
int top_index = 0;


// push the input to the stack
void push(string name) {
    myStack[++top_index] = name;
}


// pop the top in the stack and return the element being popped out
string pop() {
    return myStack[top_index--];
}


// query the element at given pos in the stack
string query(int pos) {
    return myStack[pos];
}


int main() {
    int n;
    scanf("%d", &n);
    char name[20];
    for (; n--; ) {
        int op;
        scanf("%d", &op);
        if (op == 1) {
            scanf("%s", name);
            push(name);
        } else if (op == 2) {
            printf("%s\n", pop().c_str());
        } else {
            int pos;
            scanf("%d", &pos);
            printf("%s\n", query(pos).c_str());
        }
    }
    return 0;
}

