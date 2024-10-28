# GCANet: A Geometric Consistency-driven Aggregation Network for Robust Primitive Segmentation on Point Clouds（PG2024）




## Environment
* Python 3.8
* PyTorch 1.9.0
* CUDA and CuDNN (CUDA 11.1 )
* TensorboardX (2.6) if logging training info.
* For other lib, please refer to requirements.txt
  
## Compile 
（softgroup）
``` bash
sudo apt-get install libsparsehash-dev
python setup.py build_ext develop
```
（pointnet2_ops_lib）
``` bash
cd /model/pointnet2_ops_lib
python setup.py build_ext develop
```

## Datasets
You can download the datasets used in HPNet (https://github.com/SimingYan/HPNet).



Clone this repository:
``` bash
git clone https://https://github.com/hay-001/GCANet.git
cd GCANet
```

## Train
Use the script `train_new.py` to train a model in our dataset :
(set using_set_aggr= False in option_new.py, For Training efficiently)
``` bash
cd GCANet
python train_new.py
```

## Test
(set using_set_aggr= True in option_new.py, For Training efficiently)
``` bash
cd GCANet
python train_new.py --eval
```


## Acknowledgements
This code largely benefits from following repositories:
* [HPNet](https://github.com/SimingYan/HPNet)
* [Softgroup](https://github.com/thangvubk/SoftGroup)
* [HAIS](https://github.com/hustvl/HAIS)
