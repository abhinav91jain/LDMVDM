# Learning Depth Module using DDVO

Implementation of the methods in "[Learning Depth from Monocular Videos using Direct Methods](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf)".

## Abstract
In the recent years, depth prediction from the monocular video using unsupervised methods for training have gained advances in CNNs. The paper attempts to demonstrate empirically that incorporation of recent advances in Direct Visual Odometry(DVO) with an additional CNN pose prediction model to employ a differential and deteministic approach for pose prediction substantially improves performance.

## Source Files
- **ImgPyramid.py**: To compute larger motions in DVO we form pyramid of images through downsampling.
- **MatInverse.py**: Compute inverse depth maps and camera poses as supervision to learn the depth estimator.
- **DirectVisualOdometry.py**: Objective of DVO is to find an optimum camera pose which minimizes photometric error between the warped source image and reference image(identity pose 0).
- **networks.py**: Consist of CNN models for depth estimator and Posenet.
- **KittiDataset.py**: Read and extract frames and camera parameters for each image in Kitti Dataset. 
- **CNNLearner.py**: To implement Pose predict CNN using DVO.
- **FinetuneLearner.py**: Hybrid method that uses pretrained Pose CNN with DDVO for training.

## Pre-requisites
- Python 3.6
- PyTorch 1.0.0
- Cuda 10.0
- Ubuntu 16.04

## Training

### Preparing training data
In order to train the model using the provided code, the data needs to be formatted in a certain manner. We refer "[SfMLeaner](https://github.com/tinghuiz/SfMLearner)" to prepare the training data from KITTI.
We assume the processed data is put in directory "./data_kitti/".

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command
```bash
python data/prepare_train_data.py --dataset_dir="path to raw kitti dataset" --dataset_name='kitti_raw_eigen' --dump_root="path to resulting formatted data" --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
```
### Training with different pose prediction modules
1. #### train from scratch with PoseNet
PoseNet is a pose predicting CNN which is the basic method for estimating a depth predicting CNN from monocular video.
```
bash run_train_posenet.sh
```
see [run_train_posenet.sh](https://github.com/abhinav91jain/LDMVDM/blob/master/run_train_posenet.sh) for details.

2. #### finetune with DDVO
Use pretrained posenet instead of an identity pose to give initialization for DDVO(Differential Direct Video Odometry).
```
bash run_train_finetune.sh
```
see [run_train_finetune.sh](https://github.com/abhinav91jain/LDMVDM/blob/master/run_train_finetune.sh) for details.

## Testing
- To test yourself:
```
CUDA_VISIBLE_DEVICES=0 nice -10 python src/testKITTI.py --dataset_root "Path to Kitti dataset" --ckpt_file "Path to checkpoint file" --output_path "Path to output file" --test_file_list test_files_eigen.txt
```

## Results
| Abs Rel | Sq Rel | RMSE  | RMSE(log) | d1_all | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|--------|-------|-------|-------|
| 0.4503  | 5.0245 |12.6287| 0.5968    | 0.0000 |0.2985 | 0.5495| 0.7573| 

## Acknowledgement

- Part of the code structure is borrowed from "[SfMLeaner](https://github.com/tinghuiz/SfMLearner)"
- Part of the code structure is borrowed from "[LKVOLearner](https://github.com/MightyChaos/LKVOLearner)"
