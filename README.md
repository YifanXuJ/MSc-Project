# MSc-Project - Segmentation for 4D data

## Important!

Since these data is relevant to the unpublished research, I will not release any data about this experiment, but only upload my code.

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
$ conda install -c anaconda pillow
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

## Instructions - How to use these files?
I will give detailed instructions for using my program to segment the data. You should open the scripts.sh to refer the command in it since we need to add some arguments for using these files.

### 0. Prepare the folder and environment
First, move the data which need to be segmented in the same directory with these files. Then, activate the conda virtual environment. Note that, each time, we only work on one cylinder with different time stamps. 

![Path](https://github.com/misclick47/MSc-Project/blob/master/images/path.png)

### 1. Prepare the training data
Before run the program, we should change some default parameter for our program.

#### mask_centre and radius
you should run the find_mask.py to check the centre and radius of our mask. After obtaining the correct parameter, open create_training_data.py, change the value of mask_centre and radius.

![click centre](https://github.com/misclick47/MSc-Project/blob/master/images/click_centre.png)

![show radius](https://github.com/misclick47/MSc-Project/blob/master/images/show_radius.png)


#### keyword
also, you need to change the keyword. Keywork is the name for our cylinder, like 'SHP'. Our program need this word to find all the data folder.

#### subsampling_rate
We suggest control the whole number of training data points being 2 million. You can set it freely.

![Masked image](https://github.com/misclick47/MSc-Project/blob/master/images/masked.png)

#### Run the program
We need to prepare training data for different time stamp. Refer the command in scripts.sh - Cerate training data. 

### 2. Train
In training process, we can set different parameter to train our model. See scripts - Training command.

### 3. Find label
Run analyse label.py with arguments, which will load one image, and show the segmentation results with different single labels. Then we choose the label(s) representing pores according to these images.

![raw image](https://github.com/misclick47/MSc-Project/blob/master/images/raw_image.png)

![class 0](https://github.com/misclick47/MSc-Project/blob/master/images/analyse_label_4D_0.png)
![class 1](https://github.com/misclick47/MSc-Project/blob/master/images/analyse_label_4D_1.png)
![class 2](https://github.com/misclick47/MSc-Project/blob/master/images/analyse_label_4D_2.png)

### 4. Segment
There are two files can be used to segment. segment_index.py will segment the image one by one, and it is slow.

We recommand to use segment_conv.py. It can works both on CPU and GPU. In this file, we need to set the number of slices we load together to segment. It depends on the memory of your computer. JUst in line 182, change the number of 'group'.

### 5. Transfer
Now we get 3D segmentation and 4D segmentation result saved in the corresponding time stamp folder. Assign the correct path in transfer_tif.py, and it will transfer all .png file to .tif file. Then, we can reconstruct the 3D structure by using these .tif file.





