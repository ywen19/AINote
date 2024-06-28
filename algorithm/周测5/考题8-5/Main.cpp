#include<bits/stdc++.h>

using namespace std;

// code below fully inspired by: https://blog.csdn.net/weixin_34138139/article/details/93686023

const int MAXN =  2001;
const double THRES = 1e-20;
constexpr double INF = 1E20;

typedef long long ll;


struct Point{ 
	int x,y;
	int p;  // 所在象限

	double k; // 斜率
	
    // 顺时针旋转90度
	void turn_90(){
		x=-x; 
        swap(x,y);
	}

	void calc_k(){
		if(x==0) 
			k =INF;
		else 
			k = (double)y/x;
	}
};

Point p[MAXN],  temp[MAXN];

int f[MAXN]; 
ll ans=0;

bool cmp(Point x,Point y) {return x.k < y.k;}

bool equal(double x,double y){
	if (fabs(x-y)<THRES) return 1;
    return 0;
}

int main(){
	int n,x,y; 
    scanf("%d",&n);

	for(int i=1;i<=n;i++) 
        scanf("%d%d",&p[i].x,&p[i].y);	

	for(int i=1;i<=n;i++){ 
		swap(p[i],p[n]);  // 防止把当前点再次计算坐标
		memset(temp,0,sizeof(temp));

		for(int j=1;j<n;j++){
			temp[j].x=x=p[j].x-p[n].x;
			temp[j].y=y=p[j].y-p[n].y;
			if(x>=0 && y<0){
				temp[j].turn_90();
				temp[j].turn_90();
				temp[j].turn_90();
				temp[j].p=4;
			}
			else if(x<0 && y<=0){
				temp[j].turn_90();
				temp[j].turn_90();
				temp[j].p=3;
			}
			else if(x<=0 && y>0){
				temp[j].turn_90();
				temp[j].p=2;
			}
			else temp[j].p=1;
			temp[j].calc_k();
		}

		sort(temp+1, temp+n, cmp); // 按斜率升序

		int j,k;
		for(j=1; j<n;){
			memset(f,0,sizeof(f));
			for(k=j; k<n && equal(temp[j].k,temp[k].k); k++);

			for(int t=j; t<k; t++) 
                f[temp[t].p]++;

			if(j==k) 
                j=k+1;
			else 
                j=k;

			ans += (f[1]*f[2] + f[2]*f[3] + f[3]*f[4] + f[4]*f[1]);
		}
		swap(p[n],p[i]);
	}
	printf("%lld\n",ans);
	return 0;
}