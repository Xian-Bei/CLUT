## Preparation
### Enviroment
    pip install -r requirements.txt
### Data
    Prepare the dataset in the following format and you could use the given FiveK Dataset class.
    - <data_root>
        - input_train
        - input_test
        - target_train
        - target_test
    Or you will need to implement your own class for your customed data format.

## Training
    python train.py --data_root <xx> --dataset <xx>

    The default settings of the most hyper-parameters are writen in the parameter.py file and only the data-relevent settings ('data_root' and 'dataclass') need to be modified before training.
    By default, the images, models and logs generated during training will be saved in the current directory (controled by 'save root').
### Testing
    python evaluate.py --epoch <xx>