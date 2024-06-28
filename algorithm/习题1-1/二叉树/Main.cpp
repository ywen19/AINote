#include <iostream>
#include <vector>
using namespace std;
const int N = 100005;

// 二叉树节点
// tree: 二叉树节点数组
struct node{
    int value, left, right;
}tree[N];

// root: 根节点
// cnt: 整个二叉树的大小
int root,cnt;

// 以x为根的树中插入
// v: 要插入的值
// x: 当前节点
// 返回值: x
int insert(int value, int x){
    if(x == 0){
        // 若当前节点不存在，则将x变成一个新节点
        x = ++cnt;
        tree[x].value = value;
        tree[x].left = 0;
        tree[x].right = 0;
        return x;
    }
    // 递归插入左右字数
    if(value < tree[x].value)
        tree[x].left = insert(value, tree[x].left);
    else
        tree[x].right = insert(value, tree[x].right);
    return x;
}


void dlr(int x, vector<int> &ans){
    if(x){
        ans.push_back(tree[x].value);
        dlr(tree[x].left, ans);
        dlr(tree[x].right, ans);
    }
}

// 后序遍历
// x: 当前节点
// ans: 存储结果的数组
void lrd(int x, vector<int> &ans){
    // ref: https://www.jianshu.com/p/456af5480cee
    if(x){
        lrd(tree[x].left, ans);
        lrd(tree[x].right, ans);
        ans.push_back(tree[x].value);
    }
}

// 前序遍历
vector<int> getAnswer(int n, vector<int> sequence) {
    root = cnt =0;  //初始化
    for(int i=0; i<int(sequence.size());++i)
        root = insert(sequence[i],root);
    vector<int> ans; //返回值
    dlr(root,ans);
    lrd(root,ans);
    return ans;
}

int main() {
    int n,x;
    scanf("%d", &n);
    vector<int> sequence;
    for (int i = 0; i < n; ++i) {
        scanf("%d", &x);
        sequence.push_back(x);
    }
    vector<int> ans = getAnswer(n, sequence);
    for (int i = 0; i < n; ++i)
        printf("%d%c", ans[i], " \n"[i == n - 1]);
    for (int i = 0; i < n; ++i)
        printf("%d%c", ans[n + i], " \n"[i == n - 1]);
    return 0;
}

