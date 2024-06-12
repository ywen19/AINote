# Pytorch Cuda Environment Setup

The document describes the method to set up GPU Pytorch environment with Cuda12 and Anaconda.  

## 1. Conda  
Conda is a package and environment management tool. There are similar tools such as Pyenv. From personal experience,
if were to download package, I would choose conda instead of pip, the package pulled by conda seems to be more complete
than pip.   
For work with Riot, Pyenv is the go-to choice. I might write a note someday regarding Pyenv setup.  
There are circumstances that we have to use pip. For example, within the Mayapy for Autodesk Maya, or when 
sometimes a certain package could only be pulled by pip(conda or conda-forge does not have the package).  

The download page and beginner intro for conda is: 
[Getting started with conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).  

Note that we need to add the absolute paths for conda to the environment variable(in case the installer failed to do 
so); Below image is an example:    
<img src="../image/torch_cuda/system_var.png">  

Commonly-used Commands:
* `conda create -n` create a new environment；
If you need to specify python version when creating, `conda create -n env_name python=<version>`。
Python 3.10.11 works with cuda12 and other packages best on my machine,
so the command would be `conda create -n env_name python=3.10.11`.
* `conda info --envs` list of all environments.
* `conda activate env_name` change your current environment back to specific environment.  


## 2. Cuda12 Setup  

