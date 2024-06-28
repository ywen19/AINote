#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

long long getAnswer(int n, priority_queue<ll, vector<ll>, greater<ll>> pq) {

    ll sum = 0;  // return value back to zero
    while(pq.size() > 1) {
        // if there is still element in pq
        ll newEle =  0;
        // find the smallest two elements to merge
        for (int k=0; k<2; ++k) {
            newEle += pq.top();
            pq.pop();
        }
    sum += newEle;
    pq.push(newEle);
    }
    return sum;
}



int main() {
    int n;
    ll weight;

    // priority queue
    priority_queue<ll, vector<ll>, greater<ll>> pq;

    scanf("%d", &n);

    while (n--) {
        scanf("%lld", &weight);
        pq.push(weight);
    }
    printf("%lld\n", getAnswer(n, pq));
    return 0;
}
