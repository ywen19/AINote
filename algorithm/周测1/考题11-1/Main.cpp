// 使用类似于线段树的方法，即不需要实际记录这个数组，而是记录操作和每步操作后的（历史）众数

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;  // todo: tried int but not work? Why???

// record operation and minimum major element at the given operation
struct Node {
    int x, k, majority;
};
vector<Node> seq;


// ref: https://stackoverflow.com/questions/19191247/unordered-map-what-to-return-if-key-is-not-in-map
struct CountMap : unordered_map<int, ll> {
    using unordered_map<int, ll>::operator[];

    ll operator[] (int x) const {
        // in case x not in map
        auto it = this->find(x);
        return it == this->end() ? 0 : it -> second;
    }
};
CountMap cnt;

bool compare(const CountMap& cnt, int x1, int x2) {
    // compare the passed in elements counts recorded in the count map
    // true if the first element has more frequency than the later one or same frequency but smaller value
    return cnt[x1]==cnt[x2] ? x1<x2 : cnt[x1]>cnt[x2];
}

void insert(int k, int x) {
    int majority;

    cnt[x] += k;  // update the count of element
    if (seq.empty()) {majority = x;}
    else {
        const auto &back = seq.back();
        // compare to find the current majority
        majority = compare(cnt, x, back.majority) ? x : back.majority; 
    }
    seq.emplace_back(Node{x, k, majority});
}

void remove(int k) {
    // compare k with the last operation node
    // if k is smaller than the insertion amount of the last node
    // we only need to update the last node
    // else we clear the last node and associated data
    while (k>0 && !seq.empty()) {
        auto &back = seq.back();
        if (back.k <= k) {
            k -= back.k;
            cnt[back.x] -= back.k;
            seq.pop_back();
            continue;
        }
        back.k -= k;
        cnt[back.x] -= k;
        k=0;
        if(seq.size()>1) {
            const auto& prev = seq[seq.size() - 2];
            if (compare(cnt, prev.majority, back.majority)) back.majority = prev.majority;
        }
    }
}

int main() {
    int q;
    scanf("%d", &q);

    while(q--) {
        int op;
        scanf("%d", &op);

        if (op==1) {
            int k, x;
            scanf("%d%d", &k, &x);
            insert(k, x);
        }

        else if (op==2) {
            int k;
            scanf("%d", &k);
            remove(k);
        }

        if (seq.empty()) puts("-1");
        else printf("%d\n", seq.back().majority);
    }

    return 0;
}