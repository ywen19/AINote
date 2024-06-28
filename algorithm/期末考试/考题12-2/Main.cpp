#include <bits/stdc++.h>
using namespace std;

using LL = long long;

constexpr int MAXN = 500000;

// code is inspired by: 分治求解“平面内距离最短的点对”
//https://maozezhong.github.io/2018/05/17/%E5%9F%BA%E7%A1%80_%E7%BB%8F%E5%85%B8%E7%AE%97%E6%B3%95/%E5%B9%B3%E9%9D%A2%E5%86%85%E8%B7%9D%E7%A6%BB%E6%9C%80%E7%9F%AD%E7%82%B9%E5%AF%B9/

struct Point {
    int x, y, c;
};

LL dist_sqr(const Point& a, const Point& b) {
    // 两点间距离的平方
    return pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
}

LL dist_sqr_on_x(const Point& a, const Point& b) {
    // 两点间x坐标距离的平方
    return pow(a.x - b.x, 2);
}

LL dist_sqr_on_y(const Point& a, const Point& b) {
    // 两点间y坐标距离的平方
    return pow(a.y - b.y, 2);
}


LL getMinDisttance(Point* points, int l, int r) {
    if (l >= r) return numeric_limits<LL>::max();

    int mid = (l+r) >> 1;
    LL current_dist = min(getMinDisttance(points, l, mid), getMinDisttance(points, mid+1, r));
    //cout << "current dist " << current_dist << endl;

    vector<Point> temp;  // 保存一些可能更新最短距离的点
    // 得到所有距离中心分割点小于d的坐标(注意只考虑x轴)
    for (int i=l; i<=r; ++i) {
        if (dist_sqr_on_x(points[i], points[mid]) <= current_dist) temp.push_back(points[i]);
    }

    // 按照y轴排序
    sort(temp.begin(), temp.end(), [](const Point& a, const Point& b) {return a.y<b.y;});

    // 在res中分别计算是否有距离小于d的点对，
    // 对于每个点，只需要它与距离它宽为d高为2d的矩形区域内点做对比；可以证明对于每个点，它只需要与6个点求距离
    for (int i=0; i<temp.size(); ++i) {
        for (int j=i+1; j<temp.size() && dist_sqr_on_y(temp[j], temp[i])<=current_dist; ++j) {
            if (temp[j].c != temp[i].c) {
                LL dif_color_dist = dist_sqr(temp[j], temp[i]);
                //cout << "different color dist " << dif_color_dist << endl;
                current_dist = min(current_dist, dif_color_dist);
                //cout << "after compare min dist is " << current_dist << endl;
            }
        }
    }

    return current_dist;
}


Point points[MAXN + 10];

int main() {
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) scanf("%d%d%d", &points[i].x, &points[i].y, &points[i].c);

    // 按照x坐标大小排序，分成左右做二分
    sort(points, points+n, [](const Point& a, const Point& b) {return a.x<b.x;});

    LL ans = 0;
    // 计算答案至 ans
    printf("%lld\n", getMinDisttance(points, 0, n-1));
    return 0;
}
