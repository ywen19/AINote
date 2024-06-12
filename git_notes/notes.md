# Git Notes

Here listed some common used git commands in case needed.  
A really useful tutorial is: 
[Git and GitHub Tutorial for Beginners](https://www.youtube.com/watch?v=tRZGeaHPoaw&t=1957s).

## 1 Git Setup

[Git instruction](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git)  

First download the git from the official website, then open the <b>Git Bash</b>, to configure the username and
user email.  
The commands for configuration are:
* `git init --global user.name "用户名"`
* `git init --global user.email email_address`

Below command could set up the initial branch name:
* `git config --global init.default branch main`  (main就是默认的branch名字，根据需要可变)


## 2 Set Up a Repo

### If the repo has already existed on git, we need to clone it to local machine.  
* Go to the directory where you want to put your repo in, (if first time), open <b>Git Bash</b>, 
and run `git init`. 
* `git clone` the repo from remote.

### If the project is on local PC
* Use <b>Git Bash</b> to open the project repo (command is `cd`).
* Run `git init`. If you show hidden items in your file explorer, you will find a new folder called ".git"
in your project directory. The git folder contained all the filed needed for a gir repo, such as .gitignore.
* Run `git add .` or `git add --all` to stage all files in the directory.
* Run `git commit -m "commit message"` to commit all the files.
* Run `git push`.


## 3 Some Issues While Running Section 2

* <b>Tried to push a new-cloned repo, Fatal: No configured push destination</b>  
[Solution](https://blog.csdn.net/flyingLF/article/details/104267345)  
Run `git remote add origin 'repo在git上的https(就是clone时使用的那个)'`，then `git push`.
