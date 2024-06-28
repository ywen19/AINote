#include <bits/stdc++.h>

using namespace std;

const double PI = M_PI;
const double thre = 1e-6;

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}

    void println() { printf("%.4lf %.4lf\n", (double)x, (double)y); }
};

// 先比较x轴再比较y轴，
bool operator < (const Point& a, const Point& b) {
    return a.x == b.x ? a.y < b.y : a.x < b.x;
}

// 计算a和b的叉积（外积）
double operator ^ (const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
}

// 计算a和b的点积
double operator * (const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y;
}

// 两点相减得到的向量
Point operator - (const Point& a, const Point& b) {
    return Point(a.x - b.x, a.y - b.y);
}

double distance(const Point& a, const Point& b) {
    Point v = a - b;
    return sqrt(v * v);
}

int main() {
    // 实际上要求的周长是弧度所对应的圆的周长+圆心构成的凸包的周长
    // 所以只用关注圆心即可

    double a, b, r;

    scanf("%lf%lf%lf", &a, &b, &r);
    // we can calculate the vertices for rounded rectangle (with center at (0,0))
    a = a/2 -r, b = b/2 -r;
    Point orig_center[4] = {{a, b}, {a, -b}, {-a, b}, {-a, -b}};

    int n;
    scanf("%d", &n);
    // 根据中心和旋转角变换圆心坐标
    vector<Point> centers;
    while (n--) {
        double x, y, t;
        scanf("%lf%lf%lf", &x, &y, &t);
        // 如果逆时针旋转theta, x=cos(theta)*x-sin(theta)*y, y=cos(theta)*y+sin(theta)*x
        for (const auto& center: orig_center) {
            centers.emplace_back(cos(t)*center.x-sin(t)*center.y+x, cos(t)*center.y+sin(t)*center.x+y);
        }
    }

    // 求圆心构成的凸包
    vector<Point> convex_p;
    sort(centers.begin(), centers.end());

    for (auto it=centers.begin(); it!=centers.end(); it++) {
        const auto& p = *it;
        while (convex_p.size()>1 && ((p-convex_p.rbegin()[0]) ^ (p-convex_p.rbegin()[1])) <= 0) convex_p.pop_back();
        convex_p.push_back(p);
    }

    int m = convex_p.size();
    for (auto it=centers.rbegin()-1; it!=centers.rend(); it++) {
        const auto& p = *it;
        while (convex_p.size()>=m+1 && ((p-convex_p.rbegin()[0]) ^ (p-convex_p.rbegin()[1])) <= 0) convex_p.pop_back();
        convex_p.push_back(p);
    }

    //for (int i=0; i<convex_p.size(); i++) convex_p[i].println();

    // 求解周长
    double ans = 2 * PI * r;
    for (int i=0; i<convex_p.size()-1; i++) {
        ans += distance(convex_p[i], convex_p[i+1]);
    }

    printf("%.2lf", ans);


    return 0;
}
