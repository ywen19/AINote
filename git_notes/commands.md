# Common Git Commands

1. `git status` 查看状态
2. `git branch` 列出所有的branch以及当前所在的branch
3. `git checkout` 转到某一个branch
4. `git checkout -b`   
新建一个branch; 一般来说新建后面跟`git remote add origin` 或者`git push --set-upstream origin (新建branch名)`
5. `git rebase` 
一般有两种使用情况; 
* 情景1:   
<img src="../image/git/rebase_01.png">  
举例：我们希望将从B出来的分支C和D的变化切换到master上。
注意：尽量使git history呈线性，这样更加整洁，而且例如git-p4这样的插件在非线性log的情况下会异常；
在这种情况下，运行`git checkout C`，然后运行`git rebase master`。
下图是运行以上命令后的图例：  
<img src="../image/git/rebase_02.png">  
在以上的rebase过程中，很可能会出现branch版本间的冲突；需要根据实际的命令行提示去solve conflicts;常见的命令行有：`git rebase 
--continue/skip/abort`。
记得把所有solve完的文件进行`git add .`。
小技巧：先将分支上的commits压缩(见情景2)成1个再去rebase,这样可以少solve几次conflicts。  

* 情景2：是git修改历史的一种方式
`git rebase -i Head^3`is to compress the latest 3 commits into 1;
One could also use the commit number such as `git rebase -i a5f4a0d`；
注意除了a5f4a0d是squash(s),其他commit都应该是pick(p)。

6. `git merge` 合并；一般合并前可以先做`rebase`，这样可以简化solve conflicts。完成合并后通过`git push -ff`push到remote。
7. `git push -ff`强行push;慎用!
8. `git reset`回滚；  
<img src="../image/git/reset.PNG">  
和`rebase -i`一样，可以用`Head^`也可以直接使用commit编号。
9. `git log` 查看git历史的方式
10. `git branch -D branch_name`本地删除branch, `git push remote_name(e.g. origin) -d remote_branch_name`删除remote上的分支。
11. 使用proxy时的git config(如果太慢)：
    * 首先需要注意的是不一定这样配置proxy后一定有用;
    * 假设代理是Socks5(shadowsRocket, shadowsocksR), 地址为127.0.0.1,端口为1080，则运行：  
    `git config --global https.proxy http://127.0.0.1:10800`  
    `git config --global https.proxy https://127.0.0.1:10800`  
    <mark>如果用的是http代理，则为：</mark>  
    `git config --global https.proxy http://127.0.0.1:10800`  
    `git config --global https.proxy https://127.0.0.1:10800`  
    <mark>如果代理服务器在别的主机上，例如在 192.168.1.122:8118 机器上运行代理服务程序, 运行：</mark>  
    `git config --global https.proxy http://192.168.1.122:8118`  
    `git config --global https.proxy https://192.168.1.122:8118`
    * 命令行取消proxy设置：  
    `git config --global --unset http.proxy`  
    `git config --global --unset https.proxy`  
    检查是否成功取消proxy配置:  
    `git config --global -l` 输出不应该有任何以`http.proxy=`或`https.proxy=`开头的列表元素。
    * 如果上面的命令行取消proxy设置失败了，则手动进入<mark>`C:\Users\user_name\.gitconfig`</mark>并删除掉`http.proxy=`或`https.proxy=`开头的配置。