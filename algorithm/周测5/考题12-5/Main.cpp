#include <bits/stdc++.h>

using namespace std;

using LL = long long;

struct Point { 
    int x, y; 
    void read() { scanf("%d%d", &x, &y); }
    void println() { printf("%.4lf %.4lf\n", (double)x, (double)y); }
};

Point operator- (const Point& a, const Point& b) {return Point{a.x-b.x, a.y-b.y};}
bool operator== (const Point& a, const Point& b) {return a.x==b.x && a.y==b.y;}
LL dot(const Point& a, const Point& b) {return (LL)a.x*b.x + (LL)a.y*b.y;}
LL cross(const Point& a, const Point& b) {return (LL)a.x*b.y - (LL)a.y*b.x;}


struct Segment { 
    Point p, q;
    void read() {
        p.read();
        q.read();
    }

    bool isOnSegment(const Point& P) {
        // if we have a segment from A to B
        // if dot(AC, BC)<0, C is not on the segment; else, C is on the segment
        return dot(P-p, P-q)<=0;
    }
};


int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        Segment s1, s2;
        s1.read();
        s2.read();

        // 判断交点情况，或求出具体交点

        // get the direction vec of two segments 
        Point dir1 = s1.q - s1.p, dir2 = s2.q - s2.p;
        // ref: https://blog.csdn.net/charlee44/article/details/117932408
        LL determinant = cross(dir1, dir2);

        // if and only if the cross product of two vectors is zero, they are parallel
        if (determinant==0) {
            // make the direction of two segment same for easier cconditioning(不然要写4个condition)
            if (dot(dir1, dir2) < 0) {swap(s2.p, s2.q);}
            // now s1 and s2 are of same direction
            // if s1 start is s2 end
            if (s1.p == s2.q) s1.p.println();
            // if s1 end is s2 start
            else if (s1.q == s2.p) s1.q.println();
            // if s2 start or end is on s1 segment (and is not start or end point), inf intersection points
            else if (cross(dir2, s1.p-s2.p)==0 && (s2.isOnSegment(s1.p) || s2.isOnSegment(s1.q) || s1.isOnSegment(s2.p))) {puts("inf");}
            // else no intersection
            else puts("-1");

        }
        // two segments are not parallel when the determinant is not zero
        else {
            // 联立方程： https://blog.csdn.net/Braves_wang/article/details/129315950
            // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#:~:text=By%20using%20homogeneous%20coordinates%2C%20the,w%2C%20y%2Fw).
            LL t = cross(s2.p-s1.p, dir2), u = cross(dir1, s1.p-s2.p);
            if (determinant<0) {
                determinant = -determinant;
                t = -t;
                u = -u;
            }
            if (0<=t && t<=determinant && 0<=u && u<=determinant) {
                // intersection point on segments
                double x = s1.p.x + (double)t/determinant*dir1.x, y = s1.p.y + (double)t/determinant*dir1.y;
                printf("%.4lf %.4lf\n", (double)x, (double)y);
            }
            else puts("-1");  // intersection not on segments
        }
    }
    return 0;
}
