# LDMVDM

Implementation of the methods in "[Learning Depth from Monocular Videos using Direct Methods](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf)".

## Pre-requisite
- Python 3.6
- PyTorch 1.0
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
PoseNet is a pose predicting CNN which is the basic method for the estimating a depth predicting CNN from monocular video.
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
CUDA_VISIBLE_DEVICES=0 nice -10 python src/testKITTI.py --dataset_root $DATAROOT --ckpt_file $CKPT --output_path $OUTPUT --test_file_list test_files_eigen.txt
```

## Results

## Acknowledgement

- Part of the code structure is borrowed from "[SfMLeaner](https://github.com/tinghuiz/SfMLearner)"
- Part of the code structure is borrowed from "[LKVOLearner](https://github.com/MightyChaos/LKVOLearner)"
