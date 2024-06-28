#include <bits/stdc++.h>
using namespace std;

#include<iostream>
using namespace std;

const int N = 500000;

int s[N*32+10][2],idx;

void insert(unsigned int x){
    int p = 0;
    for(int i = 31; i >= 0 ;i -- ){
        //取x从二进制上右往左第i位上的数
        int n = x >> i & 1;
        //先判断该子节点是否存在
        if(!s[p][n]){
            //若不存在的话就创建
            s[p][n] = ++ idx;
        }
        p = s[p][n];
    }
}

int query(unsigned int x){
    int res = 0;
    int p = 0;
    for(int i = 31; i>=0 ; i --){
        //取x从二进制上右往左第i位上的数
        int n = x >> i & 1;
        //先判断是否存在和n不同的子节点
        if(s[p][!n]){
            //若存在则选则此节点；并且将对应位置二进制异或的值加到结果里
            res += 1 << i;
            p = s[p][!n];
        }else{
            //若不存在则选择存在的子节点
            p = s[p][n];
        }
    }
    return res;
}


int main() {
    // n, q 含义同题目
    int n, q;
    scanf("%d%d", &n, &q);
    // x 为题目所给 n 个非负整数的数组
    while (n--) {
        unsigned int x;
        scanf("%u", &x);
        insert(x);
    }
    
    while (q--) {
        unsigned int x;
        scanf("%u", &x);
        printf("%u\n", query(x));
    }
    return 0;
}
