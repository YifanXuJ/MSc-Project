# MSc-Project - Segmentation for 4D data

## Environment building

Recommend using miniconda (if do not have Anaconda) to build the virtual environment. If you want to run all the codes, you may need to install all the package.

### Download the Miniconda
Download miniconda and create the environment:
```sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ source .bashrc
```
### Create the environment
Then we need to create a virtual environment, name it as 'segment' (or any name):
```sh
$ conda create -n segment python=3
$ conda activate segment
```
The following operations will be under 'segment' environment.

### Install necessary package
Install all the dependency package:
```sh
$ conda install -c anaconda git
$ conda install -c anaconda numpy
$ conda install -c conda-forge matplotlib
$ conda install -c anaconda scipy
$ conda install -c anaconda jupyter
$ conda install -c anaconda scikit-learn
$ conda install -c conda-forge opencv
$ conda install -c conda-forge pbzip2 pydrive
$ conda install -c conda-forge umap-learn (if you do not need to visualise in umap, do not need to install this package)
```
If you only want to run the code on CPU, you need to install tensorflow cpu version:
```sh
$ conda install -c anaconda tensorflow
```
If you want to run the code on GPU, you need to do more work.
First, open this website: https://developer.nvidia.com/cuda-downloads, and choose the correct driver to install.
Then run the following command:
```sh
$ sudo apt install nvidia-utils-418
$ pip install GPUtil
$ sudo apt install gcc
$ conda install -c anaconda cudatoolkit
$ conda install -c anaconda tensorflow-gpu
```

### Clean the cache
After successfully install all the package, run:
```sh
$ conda clean -t
```
to clean the cache.

### Quit the virtual environment
```sh
$ conda deactivate
```





