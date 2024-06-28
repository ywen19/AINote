#include <bits/stdc++.h>
using namespace std;

// inspired by: https://www.youtube.com/watch?v=Vtckgz38QHs


int partrition(vector<int> &a, int start_idx, int end_idx, int pivot) {
    // return the pivot index for partrition the passed in array

    int i = start_idx;
    int j = start_idx;

    while(i<=end_idx) {
        if (a[i] > pivot) i++;
        else {
            iter_swap(a.begin()+i, a.begin()+j);
            i++;
            j++;
        }
    }
    return j-1;
}


void quickSort(vector<int> &a, int low, int high) {
    if (high <= low) return;

    int pivot = a[high];
    int partrition_pos = partrition(a, low, high, pivot);

    quickSort(a, low, partrition_pos-1);
    quickSort(a, partrition_pos+1, high);
}


int main() {
    int n;
    scanf("%d", &n);
    vector<int> a;
    for (int i = 0; i < n; ++i) {
        int x;
        scanf("%d", &x);
        a.push_back(x);
    }
    quickSort(a, 0, n-1);
    for (int i = 0; i < n; ++i)
        printf("%d%c", a[i], " \n"[i == n - 1]);
    return 0;
}
