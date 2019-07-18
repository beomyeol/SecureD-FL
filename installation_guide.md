# GPU Cluster Setup Guide for Pytorch

## Target environment
* Ubuntu 18.04
* Python 3.6
* PyTorch v1.1
* CUDA 10.0


## Install dependencies
```
$ sudo apt install build-essential python3-pip
```

## Install CUDA 10.0

*CAVEATS: Pytorch v1.1 does not support CUDA 10.1 yet (Only CUDA 9.0 and 10.0 are supported)*

You can download the CUDA 10.0 installer for Ubuntu 18.04 [here](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804).
In this document, we are using the runfile version installer.

```
$ wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
$ sudo sh cuda_10.0.130_410.48_linux
```

Then, follow the command-line prompts. NVIDIA driver should be installed if not installed before.

By default, CUDA library is installed at `/usr/local/cuda`.

### Add CUDA libraries to LD_LIBRARY_PATH

* Option 1: User-level
  * Put the following line to `~/.bashrc` or `~/.zshrc`: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
* Option 2: System-level
  * Create a new file `cuda.conf` at `/etc/ld.so.conf.d/` and put the following line to the file: `/usr/local/cuda/lib64`

### Check GPU status
```
$ nvidia-smi
```

## Install PyTorch
On Ubuntu 18.04, the default version of Python 3 is 3.6. PyTorch has pre-built wheels for Python 3.6 and CUDA 10.0.

We are installing PyTorch via pip. You can check the [PyTorch website](https://pytorch.org/get-started/locally/) for other options.

```
$ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
$ pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

Now, PyTorch v1.1 is installed. We can check the installed PyTorch version as follows.

```
$ python3 -c "import torch; print(torch.__version__)"
```

## Miscellaneous

The federated framework uses pre-processed federated datasets that are formatted in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format). To read HDF5 in Python, the following module should be installed.

```
$ pip3 install h5py
```