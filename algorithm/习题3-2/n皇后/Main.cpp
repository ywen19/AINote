#include <bits/stdc++.h>
using namespace std;

int ans, allOne;

// pos: 二进制上的某个位置的1表示当前所在行的相应的列放了一个皇后
// left: 二进制上的某个位置的1表示当前所在行的相应的位置（即右对角线上已有皇后），不能放置皇后
// left: 二进制上的某个位置的1表示当前所在行的相应的位置（即左对角线上已有皇后），不能放置皇后
void dfs(int pos, int left, int right) {
	if (pos==allOne) {
		// 当且仅当每一列都放了一个皇后，那么整个键盘已经合法放了n个皇后，则终止
		++ans;
		return;
	}

	for (int t= allOne & (~(pos|left|right)); t;) {
		// t代表可以放的集合对应的二进制数
		int p = t & -t;  // low bit: 二进制中最右边的1
		dfs(pos|p, (left|p)<<1 & allOne, (right|p)>>1);
		t ^= p; // 消除low bit

	}
}

/* 请在这里定义你需要的全局变量 */

// 一个n×n的棋盘，在棋盘上摆n个皇后，求满足任意两个皇后不能在同一行、同一列或同一斜线上的方案数
// n：上述n
// 返回值：方案数
int getAnswer(int n) {
    ans = 0;
	allOne = (1<<n) -1;
	dfs(0, 0, 0);
	return ans;
}

// ================= 代码实现结束 =================

int main() {
    int n;
	cin >> n;
	cout << getAnswer(n) << endl;
	return 0;
}
