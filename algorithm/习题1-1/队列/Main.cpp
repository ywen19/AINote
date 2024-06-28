#include <bits/stdc++.h>
using namespace std;
#define MAX_SIZE 100000

class Queue{
    string stackArr[MAX_SIZE];
    int head, tail = 0;

    public:
        Queue() {head, tail = 0;}

        void enqueue(string name) {
            stackArr[tail++] = name;
        }

        string dequeue() {
            return stackArr[head++];
        }

        string query(int pos) {
            return stackArr[head+pos-1];
        }

};

Queue myQueue;

int main() {
    int n;
    scanf("%d", &n);
    char name[20];
    for (; n--; ) {
        int op;
        scanf("%d", &op);
        if (op == 1) {
            scanf("%s", name);
            myQueue.enqueue(name);
        } else if (op == 2) {
            printf("%s\n", myQueue.dequeue().c_str());
        } else {
            int pos;
            scanf("%d", &pos);
            printf("%s\n", myQueue.query(pos).c_str());
        }
    }
    return 0;
}
