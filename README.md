# GCANet: A Geometric Consistency-driven Aggregation Network for Robust Primitive Segmentation on Point Clouds




## Environment
* Python 3.8
* PyTorch 1.9.0
* CUDA and CuDNN (CUDA 11.1 )
* TensorboardX (2.6) if logging training info. 

## Datasets
You can download the datasets used in HPNet (https://github.com/SimingYan/HPNet).



Clone this repository:
``` bash
git clone https://https://github.com/hay-001/GCANet.git
cd GCANet
```

## Train
Use the script `train_new.py` to train a model in our dataset :
``` bash
cd GCANet
python train_new.py
```

## Test
``` bash
cd GCANet
python train_new.py --eval
```


```



## Acknowledgements
This code largely benefits from following repositories:
* [HPNet](https://github.com/SimingYan/HPNet)
* [Softgroup](https://github.com/thangvubk/SoftGroup)
* [HAIS](https://github.com/hustvl/HAIS)
