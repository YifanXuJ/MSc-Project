# MSc-Project - Segmentation for 4D data

## Install

Recommend using miniconda to build the environment.

Download miniconda and create the environment:
```sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ source .bashrc
$ conda create -n segment python=3
$ conda activate segment
```

Then install all the dependency package:
```sh
$ conda install -c anaconda git
$ conda install -c anaconda numpy
$ conda install -c conda-forge matplotlib
$ conda install -c anaconda scipy
$ conda install -c anaconda pandas
$ conda install -c anaconda jupyter
$ conda install -c anaconda scikit-learn
$ conda install -c conda-forge opencv
$ conda install -c conda-forge pbzip2 pydrive
```
If just want to run the code on CPU, need to install tensorflow cpu version:
```sh
$ conda install -c anaconda tensorflow
```


Then, install the GPU-related package:
Open this website: https://developer.nvidia.com/cuda-downloads, and choose the correct driver.
Then run the following command:
```sh
$ sudo apt install nvidia-utils-418
$ pip install GPUtil
$ sudo apt install gcc
$ conda install -c anaconda cudatoolkit
$ conda install -c anaconda tensorflow-gpu
```

Finally, run:
```sh
$ conda clean -t
```
