#include <bits/stdc++.h>
using namespace std;

typedef long long ll;


bool check(ll sum_d, int n, int m, vector<int> &a) {
    /*
    Check if array could be splitted within m partritions and each partrition sum is smaller then the upper bound;
    sum_d: each partition sum should not exceed sum_d
    n: size of a; m: amount of partitions
    a: vector to divide
    */

   ll sum = 0;  // sum from previous partrition
   int cnt = 1;  // minimum partrition amount to make sure sum does not go over sum_d

   for(int i=0; i<n; ++i) {
        if (sum_d < a[i]) return false;

        sum += a[i];
        // if the sum exceeds d, make a[i] the start of a new partrition
        if (sum_d < sum) {
            sum = a[i];
            cnt++;
        } 
   }

   return cnt<=m;
}


ll getAnswer(int n, int m, vector<int> a) {

    ll low = 1, upper = 0;  // if upper starts with 0, result for all-zero array will not be correct

    for (int i=0; i<n; ++i) {upper += a[i];}

    // dichotomy
    while (low <= upper) {
        ll mid = (low + upper)>>1;
        if(check(mid, n, m, a)) {upper = mid -1;}
        else low = mid + 1;
    }

    return upper+1;
}


int main() {
    int n, m;
    scanf("%d%d", &n, &m);
    vector<int> a;
    a.resize(n);
    for (int i = 0; i < n; ++i)
        scanf("%d", &a[i]);
    printf("%lld\n", getAnswer(n, m, a));
    return 0;
}

