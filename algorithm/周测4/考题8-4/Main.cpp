#include <bits/stdc++.h>

using namespace std;

const int MAXL = 1e6 + 5;

int ch[MAXL][26], sz[MAXL], idx;

void insert(char* s) {
    int len = strlen(s), p = 0, l=0;

    for (int i=0; i<len; i++) {
        int n = s[i] - 'a';
        if (!ch[p][n]) {
            ch[p][n] = ++idx;
        }
        p = ch[p][n];
        l++;
    }
    sz[p] = l;
}

int query(char* s) {
    int len = strlen(s), p = 0, ans = 0;

    for (int i=0; i<len; i++) {
        int n = s[i] - 'a';
        if (!ch[p][n]) return max(ans, sz[p]);

        p = ch[p][n];
        ans = max(ans, sz[p]);
    }
    return ans;
}



char s[100];
int main() {

    int n, q;
    scanf("%d%d", &n, &q);

    while(n--) {
        scanf("%s", s);
        insert(s);
    }

    sz[0] = 0;
    //ofstream myfile;
    //myfile.open ("example.txt");
    while(q--) {
        scanf("%s", s);
        //myfile << query(root, s) << endl;
        printf("%d\n", query(s));
    }
    //myfile.close();
    return 0;
}
