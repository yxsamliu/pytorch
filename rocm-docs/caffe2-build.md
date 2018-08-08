# Caffe2: Building From Source on ROCm Platform

## Intro
This instruction provides a starting point to build caffe2 on AMD GPUs (Caffe2 ROCm port) from source. This requires you to install ROCm stack on your bare metal. Install ROCm stack following steps at [link](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md) if your machine doesn't have ROCm already.

Once the machine is ready with ROCm stack, there are three ways to use caffe2 
* Run the docker container with caffe2 installed in it.

* Build caffe2 from source inside a docker environment.

* Build caffe2 from source on bare metal environment

If you are looking for docker based installations, follow the steps [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce) to install docker if you do not have one already.

## 1. Launch docker container with caffe2 pre-installed
```
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video rocm/caffe2:rocm1.8.2
```

To run benchmarks, skip directly to benchmarks section of the document.

## 2. Build caffe2 from source inside docker
### Pull the docker image
```
docker pull rocm/caffe2:unbuilt-rocm1.8.2
```
This docker image has all the dependencies for caffe2 pre-installed.

### Pull the latest caffe2 source:
* Using https 
```
git clone --recurse-submodules https://github.com/ROCmSoftwarePlatform/pytorch.git
```
* Using ssh
```
git clone --recurse-submodules git@github.com:ROCmSoftwarePlatform/pytorch.git
```
Navigate to repo directory
```
cd pytorch
```

### Launch the docker container
```	
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video -v $PWD:/pytorch rocm/caffe2:unbuilt-rocm1.8.2
``` 
Navigate to pytorch directory `cd /pytorch` inside the container.

Run the below command to build caffe2  

`.jenkins/caffe2/amd/build_amd.sh`

	
## 3. Build caffe2 from source on bare metal

Clone this repository and isntall the dependencies required by running
```
<pytorch_home>/.jenkins/caffe2/amd/install_dependecies_amd.sh
```
Run the build script to build caffe2 from source.
```
<pytorch_home>/.jenkins/caffe2/amd/build_amd.sh
```


## Test installation

Run this to see if your Caffe2 installation was successful

```
cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```
## Run benchmarks

Navigate to build directory, `cd /pytorch/build_caffe2` to run benchmarks.

Caffe2 benchmarking script supports the following networks.
1. MLP
2. AlexNet
3. OverFeat
4. VGGA
5. Inception
6. Inception_v2
7. Resnet50 

*Special case:* Inception_v2 and Resnet50 will need their corresponding protobuf files to run the benchmarks. Protobufs can be downloaded from caffe2 model zoo using the below command. Substitute model_name with `inception_v2` or `resnet50`

```
python caffe2/python/models/download.py <model_name>
```
This will download the protobufs to current working directory.

To run benchmarks for networks MLP, AlexNet, OverFeat, VGGA, Inception, run the command replacing `<name_of_the_netwrok>` with one of the networks. 

```
python caffe2/python/convnet_benchmarks.py --batch_size 64 --model <name_of_the_network> --engine MIOPEN --layer_wise_benchmark True --net_type simple

```
To run Inception_v2 or Resnet50, please add additional argument `--model_path` to the above command which should point to the model directories downloaded above.

```
python caffe2/python/convnet_benchmarks.py --batch_size 64 --model <name_of_the_network> --engine MIOPEN --layer_wise_benchmark True --net_type simple --model_path <path_to_model_protobufs>

```

