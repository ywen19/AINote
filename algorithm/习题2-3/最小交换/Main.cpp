#include <bits/stdc++.h>
using namespace std;

// seq: original sequence, seqTemp: support sequence 
// cnt: amount of inverse pairs
typedef long long ll;

vector<int> seq, seqTemp;
ll cnt;

void mergeSort(int l, int r) {
    if (r<=l) return;

    int mid = (l+r)/2;
    mergeSort(l, mid);
    mergeSort(mid+1, r);

    // two pointer to compare two elements
    int p = l, q = mid + 1;
    for (int i=l; i<=r; ++i) {
        if (q>r || p <= mid && seq[p] <= seq[q]) {
            seqTemp[i] = seq[p++];  // if left element is smaller than right element
        }
        else {
            seqTemp[i] = seq[q++];  // if right element is smaller than left element
            cnt += mid - p + 1;
        }
    }

    for (int i=l; i<=r; ++i) {
        seq[i] = seqTemp[i];  // temp to the original sequence
    }
}



int main() {
    int n, tmp;
    vector<int> a;
    a.clear();
    scanf("%d", &n);
    for (int i = 1; i <= n; ++i) {
        scanf("%d", &tmp);
        a.push_back(tmp);
    }
    seq = a;
    seqTemp.resize(n);
    cnt = 0;

    mergeSort(0, n-1);
    cout << cnt << '\n';
    return 0;
}
