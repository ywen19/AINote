#include <iostream>
#include <string>
#include <cstdio>
#include <algorithm>

#define MAXN 5000

using namespace std;

// Needleman Wunsch algorithm ref: https://blog.csdn.net/qq_54501547/article/details/127863580
int match = 4, miss = 2, gap = 1;

// a: store the int id for the input characters in string a
// b: store the int id for the input characters in string b
char a[MAXN + 10], b[MAXN + 10];
int na, nb;

int score[MAXN + 10][MAXN + 10];

int main() {
    scanf("%d%s", &na, a + 1);
    scanf("%d%s", &nb, b + 1);

    // score matrix 
    // 初始化边缘分, a在行, b在列
    for (int i=1; i<=na; i++) score[i][0] += gap*i;
    for (int i=1; i<=nb; i++) score[0][i] += gap*i;

    // valculate score
    for (int i=1; i<=na; i++) {
        for (int j=1; j<=nb; j++) {
            int left = score[i-1][j] + gap;
            int above = score[i][j-1] + gap;
            // diagonal
            int diagonal = score[i-1][j-1];
            if (a[i] == b[j]) diagonal += match;
            else diagonal += miss;
            score[i][j] = max(left, max(above, diagonal));
        }
    }


    printf("%d\n", score[na][nb]);
    return 0;
}
