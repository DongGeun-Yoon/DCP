# Matting Prune

  This repository is official pytorch implementation of the paper "[Lightweight Alpha Matting Network Using Distillation-Based Channel Pruning](https://arxiv.org/abs/2210.07760)" (ACCV 2022) which is a follow-up research of the paper of [SPKD](https://github.com/DongGeun-Yoon/SPKD) (IEEE SPL 2020).
 
 Donggeun Yoon, Jinsun Park, Donghyeon Cho
 
## Performace
### note
Here is the results of DIM-student with and without knowledge distillation on the Adobe Image Matting Dataset:
|Methods|MSE|SAD|Grad|Conn|#Param|FLOPs
|---|---|---|---|---|---|---|
|Teacher|0.021|65.37|67.58|25.58|25.58M|24.51G|
|UNI|0.049|114.02|78.89|122.00|6.40M|6.19G|
|NS|0.052|120.82|83.10|129.70|6.32M|7.65G|
|CAP|0.040|101.77|63.93|108.61|4.08M|4.26G|
|NST|0.033|0.038|56.36|95.20|5.98M|11.23G|
|OFD|0.032|76.71|43.86|80.61|7.97M|17.61G|
|SPKD|0.027|73.67|40.78|76.58|7.81M|12.53G|

## Prepare
### Dataset
1. Please contact authors requesting for the Adobe Image Matting dataset.
2. Download images from the COCO and Pascal VOC datasets in folder `data` and Run the following command to composite images.  
```bash
$ python pre_process.py
```
3. Run the following command to seperate the composited datasets with training set and valid set.
```bash
$ python data_gen.py
```

### Pre-trained model
Download pretrained [teacher model](https://github.com/foamliu/Deep-Image-Matting-PyTorch) before train and place in folder `pretrained`.

## Pruning Stage
Our method consists of two stages: Pruning stage and Training Stage.
First, Run the following command to get pruned model suitable for SPKD. 

```bash
$ python train_prune.py --config configs/train_SPKD.yaml
```

## Training Stage
Second, Train pruned model. The model cfg is saved in `result/SPKD/pruned.tar`
```bash
$ python train_pruned.py --config configs/train_SPKD.yaml
```

## Citation

```
@article{yoon2022lightweight,
  title={Lightweight Alpha Matting Network Using Distillation-Based Channel Pruning},
  author={Yoon, Donggeun and Park, Jinsun and Cho, Donghyeon},
  journal={arXiv preprint arXiv:2210.07760},
  year={2022}
}
```

## Acknowledgement
The code is built upon [Deep image matting (pytorch)](https://github.com/foamliu/Deep-Image-Matting-PyTorch).