#include <iostream>
#include <vector>

struct Node {
    int index;
    Node *prev;
    Node *succ;
};
std::vector<Node> dLinkList;

void init(int n) {
    dLinkList.resize(n+1);
    for (int i=0; i<n+1; ++i) {
        dLinkList[i].index = i;
        dLinkList[i].prev = NULL;
        dLinkList[i].succ = NULL;
    }
}

bool split_succ(int x) {
    if (dLinkList[x].succ==NULL) return false;

    Node *succ = dLinkList[x].succ;
    if (succ->prev!=NULL) succ->prev = NULL;
    dLinkList[x].succ = NULL;
    return true;
}

bool split_prev(int x) {
    if (dLinkList[x].prev==NULL) return false;

    Node *prev = dLinkList[x].prev;
    if (prev->succ!=NULL) prev->succ = NULL;
    dLinkList[x].prev = NULL;
    return true;
}

bool link(int x, int y) {
    if (dLinkList[x].succ!=NULL || dLinkList[y].prev!=NULL) return false;

    dLinkList[x].succ = &dLinkList[y];
    dLinkList[y].prev = &dLinkList[x];
    return true;
}


std::vector<int> visit_succ(int x) {
    std::vector<int> ans;
    ans.push_back(x);

    Node *head = &dLinkList[x];

    while (head->succ!=NULL) {
        //std::cout << "head index is: " << head->succ << std::endl;
        if (head->succ == &dLinkList[x]) 
            break;

        ans.push_back(head->succ->index);
        //std::cout << "succ index is: " << head->succ->index << std::endl;
        head = head->succ;
        //std::cout << "head index is: " << head->index << std::endl;
    }
    return ans;
}

// x 是编号，范围为 1 到 n
// 返回遍历得到的序列
std::vector<int> visit_prev(int x) {
    std::vector<int> ans;
    ans.push_back(x);

    Node *head = &dLinkList[x];

    while (head->prev!=NULL) {
        if (head->prev == &dLinkList[x]) break;

        ans.push_back(head->prev->index);
        head = head->prev;
    }
    return ans;
}


// 代码实现结束

int main() {
    std::ios::sync_with_stdio(false);
    int n, m, x, y;
    std::string op;
    std::cin >> n >> m;
    init(n);
    for (int i = 0; i < m; ++i) {
        std::cin >> op >> x;
        if (op == "split_succ") {
            std::cout << (split_succ(x) ? "yes" : "no") << '\n';
        } else if (op == "split_prev") {
            std::cout << (split_prev(x) ? "yes" : "no") << '\n';
        } else if (op == "link") {
            std::cin >> y;
            std::cout << (link(x, y) ? "yes" : "no") << '\n';
        } else if (op == "visit_succ") {
            std::vector<int> ans = visit_succ(x);
            for (int i = 0; i < ans.size(); i++) {
                std::cout << ans[i] << " \n"[i + 1 == ans.size()];
            }
        } else {
            std::vector<int> ans = visit_prev(x);
            for (int i = 0; i < ans.size(); i++) {
                std::cout << ans[i] << " \n"[i + 1 == ans.size()];
            }
        }
    }
    return 0;
}
