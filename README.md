# Unlocking the Potential of LDPC Decoders with PiM Acceleration


## Introduction

This repo contains the code and results for paper to be published in the MDPI Remote Sensing.

If you any of this work in your research, please cite:



	 @article{ferraz2021hyperspectral,
    title={Hyperspectral parallel image compression on edge GPUs},
    author={Ferraz, Oscar and Silva, Vitor and Falcao, Gabriel},
    journal={Remote Sensing},
    volume={13},
    number={6},
    pages={1077},
    year={2021},
    publisher={MDPI}
    }
    
O. Ferraz, V. Silva and G. Falcao, "Hyperspectral parallel image compression on edge GPUs," Remote Sensing 2021.

<!--S. Subramaniyan et al., "Enabling High-Level Design Strategies for High-Throughput and Low-power NB-LDPC Decoders," in IEEE Design & Test, 2022, doi: 10.1109/MDAT.2022.3202852. https://ieeexplore.ieee.org/document/9869892
Please cite my work if this code or this papers are useful for you.

I might release a cleaner version of this some time in the future, but probably not, because I'm working on other stuff now.

Good luck!-->

<!--## Networks and hardware

The following networks were used in this work:

    - AlexNet
    - GoogleNetV1
    - SqueezeNet
    - ResNet-18
    - ResNet-50
    - ResNet-101
    - ResNet-152
    - VGG-16
    - VGG-19

All of these networks can be found at https://drive.google.com/drive/folders/1rdBGUq4RYoNh70cXGpIehNTI-j27ZM6g?usp=sharing

The inference was executed in:

    - Intel Movidius neural compute stick (NCS)
    - Nvidia Jetson Nano
    - Nvidia Jetson TX2
    - Nvidia Jetson Xavier

## How to run

### NCS

The ncappzoo folder contains the files to run the inferences in the NCS. To compile the model, run the makefile at ncappzoo/networks/*network* where *network* is the selected network. These makefiles have option to select the precision. These makefiles will run a simple classifier producing the accuracy for a defined set of images.

To measure performance, copy the compiled networks (*.xml and *.bin) from the previous step to ncappzoo/apps/benchmark_ncs/ and run the makefile. The makefile might need some configuration to change the name of the model and the used images. The number of inferences, threads per device, and simultaneous inferences per thread can be configure in the benchmark_ncs.py file. The results for throughput used number_of_inferences = 20, threads_per_dev = 10, simultaneous_infer_per_thread = 2.

### Nvidia Jetson

#### Accuracy

The jetson_inference_* folder contains the files to run the inferences in the Jetson boards. To run the inference run, run the docker container:

    $ cd jetson-inference
    $ docker/run.sh

For accuracy tests, modify the c/imageNet.h file and change the precison the 2nd static imageNet* Create function. Save and compile the new file:

    $ cd build
    $ cmake ..
    $ make all

Next, copy the file data/run.sh and paste at build/aarch/bin/ and run the file:

    $ cp data/run.sh build/aarch/bin/
    $ cd build/aarch/bin/
    $ bash run.sh

To run the throughput tests run the following:

    $ cd data
    $ bash run_infer.sh

To run the power tests run the following:

    $ cd data
    $ bash power2.sh

## Takeaways

- Perfomance
	- Using reduced precision (FP16) doubles throughput performance in GPUs (Jetson Nano, Jetson TX2, and Jetson AGX Xavier) 
	- Using reduced precision (FP16) increases efficiency in GPUs (Jetson Nano, Jetson TX2, and Jetson AGX Xavier) 
	- No noticeable differences were observed between using full precision (FP32) and reduced precision (FP16) in the NCS and DLAs for both throughput and efficiency. 
	- There is an apparent link between the number of parameters of the network and the throughput performance (+ number of parameters, - throughput)

- Power Modes
	- The NCS has the similiar efficiency as the Jetson TX2 but only requires 1.8W
	- DLAs have the best efficiency for full precision (FP32)
	- GPUs scale better using reduced precision (FP16)
![Alt text](https://github.com/oscarferraz/ICASSP23/blob/master/Throughput2.svg)
![Alt text](https://github.com/oscarferraz/ICASSP23/blob/master/efficiency3.svg)
