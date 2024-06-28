#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

const int Mod = 1000003;
vector<ll> hash_table[Mod];

// Hash function
int Hash(ll x) {
    return x % Mod;
}


bool check(int op, ll x) {
    int h = Hash(x);  // find the hash value for x

    vector<ll>::iterator ptr = hash_table[h].end();  // ietrator for vector

    // find x in the hash table
    for (vector<ll>::iterator it = hash_table[h].begin(); it != hash_table[h].end(); ++it) {
        if (*it == x) {
            ptr = it;
            break;  // break if find
        }
    }

        if (op == 1) {
            // if operation is 1, insert 
            if (ptr == hash_table[h].end()) {
                hash_table[h].push_back(x); 
                return 1;  // if not find in hash table, add the value to the table
            }
            return 0;
        }

        else {
            // if operation is 2, delete
            if (ptr != hash_table[h].end()) {
                *ptr = hash_table[h].back();
                hash_table[h].pop_back();
                return 1;
            }
            return 0;
        }
}


int main() {
    int Q, op;
    ll x;
    scanf("%d", &Q);
    while (Q--) {
        scanf("%d%lld", &op, &x);
        puts(check(op, x) ? "Succeeded" : "Failed");
    }
    return 0;
}
