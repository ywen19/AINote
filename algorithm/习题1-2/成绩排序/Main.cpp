#include <bits/stdc++.h>
using namespace std;

/* 请在这里定义你需要的全局变量 */

// 这是进行排序的函数
// n：题目描述中的 n
// A：各同学的算法训练营成绩
// DS：各同学的数据结构训练营成绩
// 返回值：将要输出的数字依次加入到返回值的数组中
vector<int> getAnswer(int n, vector<int> A, vector<int> DS) {
    vector<int> sum, id;  // total score and student id
    vector<int> ans;  // return value

    for (int i=0; i<n; ++i) {
        id.push_back(i+1);
        sum.push_back(A[i] + DS[i]);  // sum of the scores
    }

    int cnt = 0;  // record amount of inverse pairs

    // bubble sort in descending order
    // ref: https://www.javaguides.net/2023/09/c-program-bubble-sort-in-descending-order.html
    for (int i=0; i<n-1; ++i) {
        for (int j=0; j<n-i-1; j++) {
            // if current student has higher total score or if has the same total score but higher A score
            if (sum[j]<sum[j+1] || (sum[j+1]==sum[j] && A[j]<A[j+1])) {
                //swap
                swap(id[j], id[j+1]);
                swap(sum[j], sum[j+1]);
                swap(A[j], A[j+1]);
                swap(DS[j], DS[j+1]);
                // add 1 to inverse pair count
                cnt += 1;
            }
        }
    }

    // organize the return value
    for (int i=0; i<n; ++i) {
        ans.push_back(id[i]);
        ans.push_back(sum[i]);
        ans.push_back(A[i]);
        ans.push_back(DS[i]);
    }

    // add count of inverse pairs to the ans vector
    ans.push_back(cnt);
    return ans;
}


int main() {
    int n;
    scanf("%d", &n);
    vector<int> A, DS;
    for (int i = 0; i < n; ++i) {
        int a, ds;
        scanf("%d%d", &a, &ds);
        A.push_back(a);
        DS.push_back(ds);
    }
    vector<int> ans = getAnswer(n, A, DS);
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
        printf("%d %d %d %d\n", ans[cnt], ans[cnt + 1], ans[cnt + 2], ans[cnt + 3]);
        cnt += 4;
    }
    printf("%d\n", ans[cnt]);
    return 0;
}
