#include <iostream>
#include <vector>
using namespace std;

struct Node {
    Node* left = NULL;
    Node* right = NULL;
};


bool identicalTrees(Node* a, Node* b)
{
    /*1. both empty */
    if (a == NULL && b == NULL)
        return true;
 
    /* 2. both non-empty -> compare them */
    if (a != NULL && b != NULL) {
        return (identicalTrees(a->left, b->left)
                && identicalTrees(a->right, b->right));
    }
 
    /* 3. one empty, one not -> false */
    return false;
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    int T;
    cin >> T;
    while (T--) {
        int n;
        Node tree_A[n];
        Node tree_B[n];

        cin >> n;
        for (int i = 0; i < n; i++) {
            int l, r;
            cin >> l >> r;
            //a.push_back((node){l, r});
            if (l!=-1) tree_A[i].left = &tree_A[l];
            if (r!=-1) tree_A[i].right = &tree_A[r];
        }
        for (int i = 0; i < n; i++) {
            int l, r;
            cin >> l >> r;
            //b.push_back((node){l, r});
            if (l!=-1) tree_B[i].left = &tree_B[l];
            if (r!=-1) tree_B[i].right = &tree_B[r];
        }

        cout << (identicalTrees(&tree_A[0], &tree_B[0]) ? "yes" : "no") << '\n';
    }

    return 0;
}