# Core codes for CLUT-Net and tools for 3DLUT
[**CLUT-Net: Learning Adaptively Compressed Representations of 3DLUTs for Lightweight Image Enhancement**](/demo_images/MM2022%20CLUT-Net.pdf)

Fengyi Zhang, [Hui Zeng](https://huizeng.github.io/), [Tianjun Zhang](https://github.com/z619850002), [Lin Zhang](https://cslinzhang.gitee.io/home/)

*ACMMM2022* 

## Overview
![](/demo_images/overview.png)
Framework of our proposed CLUT-Net which consists of 
- a neural network
- *N* basis CLUTs
- two transformation matrices

The *N* basis CLUTs cover various enhancement effects required in different scenes. The neural network predicts content-dependent weights according to the downsampled input to fuse the basis CLUTs into an image-adaptive one, from which the transformation matrices adaptively reconstruct the corresponding standard 3DLUT to enhance the original input image. 

All three modules are jointly learned from the annotated data in an end-to-end manner.
## Preparation
### Enviroment
    pip install -r requirements.txt

The fast deployment of 3DLUT relies on the CUDA implementation of trilinear interpolation in [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

To install their **trilinear** library: 

    cd trilinear_cpp
    sh setup.sh

### Data
- [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek/) & [HDR+ Burst Photography Dataset](http://www.hdrplusdata.org/)
    - We use the setting of [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) in our experiments, please refer to their page for details and data link.
- [PPR10K](https://github.com/csjliang/PPR10K)

Prepare the dataset in the following format and you could use the provided [FiveK Dataset class](/datasets.py).

    - <data_root>
        - input_train
        - input_test
        - target_train
        - target_test

Or you need to implement your own Class for your customed data format / directory arrangement.

## Training
The default settings of the most hyper-parameters are written in the [parameter.py](parameter.py) file.
To get started as soon as possible (with the FiveK dataset), only the 'data_root' needs to be modified before training.

    python train.py --data_root <path>

By default, the images, models, and logs generated during training are saved in [save_root/dataset/name](/FiveK/20%2B05%2B10_models/).
## Evaluation
We provide two pretrained models on the FiveK datset:
    
  - 20+05+10: 25.56 PSNR
  - 20+05+20: 25.58 PSNR 

Just
    
    python evaluate.py --model 20+05+10 --epoch 305
or

    python evaluate.py --model 20+05+20 --epoch 310

To evaluate your trained model of a specific epoch, remember to keep the other parameters the same as training.



## Visualization & Analysis
- Strong correlations 
![](demo_images/S.png)
    
- Weak correlations 
![](demo_images/W.png)

- Learned matrices
![](demo_images/matrix_W.png)

- 3D visualization of the learned basis 3DLUTs **(Left: initial identity mapping. Right: after training)**
![](demo_images/3D.png)
![](demo_images/3D_2.png)

All the visualization codes could be found in [utils/](./utils/).

# Acknowledgemen
Thanks to Zeng *et al* for their excellent work.
This repo is built on their implementation:
[Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time *TPAMI2020*](https://github.com/HuiZeng/Image-Adaptive-3DLUT)

Hope our work helps.




    