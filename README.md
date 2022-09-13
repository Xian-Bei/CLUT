# Core codes for CLUT-Net and tools for 3DLUT
CLUT-Net: Learning Adaptively Compressed Representations of 3DLUTs for Lightweight Image Enhancement

*ACMMM2022* 
## Preparation
### Enviroment
    pip install -r requirements.txt
### Data
Prepare the dataset in the following format and you could use the provided FiveK Dataset class.

    - <data_root>
        - input_train
        - input_test
        - target_train
        - target_test

Or you need to implement your own Class for your customed data format / directory arrangement.

## Training
    python train.py --data_root <xx> --dataset <xx> 

The default settings of the most hyper-parameters are written in the *parameter.py* file.
To get started as soon as possible, only the 'data_root' needs to be modified before training.
By default, the images, models, and logs generated during training will be saved in the parent dir of the current one (controlled by 'save root').
## Testing
    python evaluate.py --epoch <xx>
To evaluate the trained model of a specific epoch, keep the other parameters the same as training.


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

All the visualization codes could be found in the *utils* directory.
# Acknowledgement
Our work is built on the excellent work of Zeng *et al*.

[Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time](https://github.com/HuiZeng/Image-Adaptive-3DLUT)
*TPAMI2020*

    