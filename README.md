# CoSeg

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/CoSeg-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-MIT-green">
</a>   

CoSeg is a self-supervised learning-based event boundary detection method.

Copyright (C) 2022 Xiao Wang, Jingen Liu, Tao Mei, Jiebo Luo

License: MIT for academic use.

Contact: Xiao Wang (wang3702@purdue.edu), Jingen Liu (jingenliu@gmail.com)

## Introduction

Some cognitive research has discovered that humans accomplish event segmentation as a side effect of event anticipation. 
Inspired by this discovery, we propose a simple yet effective end-to-end self-supervised learning framework for event
segmentation/boundary detection. Unlike the mainstream clustering-based methods, our framework exploits a transformer-based
feature reconstruction scheme to detect event boundary by reconstruction errors. This is consistent with the fact that humans spot new
events by leveraging the deviation between their prediction and what is actually perceived. Thanks to their heterogeneity in semantics,
the frames at boundaries are difficult to be reconstructed (generally with large reconstruction errors), which is favorable for event
boundary detection. Additionally, since the reconstruction occurs on the semantic feature level instead of pixel level, we develop a
temporal contrastive feature embedding module to learn the semantic visual representation for frame feature reconstruction. This
procedure is like humans building up experiences with “long-term memory”. The goal of our work is to segment generic events rather
than localize some specific ones. We focus on achieving accurate event boundaries. As a result, we adopt F1 score (Precision/Recall)
as our primary evaluation metric for a fair comparison with previous approaches. Meanwhile, we also calculate the conventional
frame-based MoF and IoU metric. We thoroughly benchmark our work on four publicly available datasets and demonstrate much better
results.

## Installation
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:wang3702/CoSeg.git && cd CoSeg
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip install torch==1.8.1
pip install torchvision==0.9.1
pip install Pillow
pip install opencv-python
pip install scipy
pip install tensorboardX
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n CoSeg python=3.7.1
conda activate CoSeg
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate CoSeg
conda deactivate(If you want to exit) 
```

#### 4 Prepare Datasets
Please search Breakfast, 50 Salads, INRIA dataset online and you can easily get access to download them.

## Usage

### 1 Frame Data Preparation
```
python3 main.py --mode=0 -F=[video_dataset_dir] -M=[frame_dataset_dir] --dataset=[dataset_type]
```
[video_dataset_dir] is the unzipped directory of your downloaded dataset, which should includes many video files inside;
[frame_dataset_dir] is the specified directory to save the generated frames, which will be used for our training and inference;
[dataset_type] specifies the dataset name, "B" indicates Breakfast dataset, "S" indicates the 50 Salad dataset, "I" indicates the INRIA dataset.

### 2 Training of CoSeg

```
python3 main.py --mode=1 -F=[frame_dataset_dir] --dataset=[dataset_type] \
 --epochs [epochs] --batch_size [batch_size] --lr [learning_rate] \
 --gpu [gpu_id] --log_path [log_dir] --window_length [window_size] \
 --mask_length [mask_size] --beta [loss_coefficient] --tensorboard [tensorboard_flag]
```
[frame_dataset_dir] is the specified directory to save the generated frame, which is generated in the previous step.
[dataset_type]  specifies the dataset name, "B" indicates Breakfast dataset, "S" indicates the 50 Salad dataset, "I" indicates the INRIA dataset.
[epochs] is the number of epochs for training. [batch_size] is the training batch size. 
[learning_rate] is the learning rate for training the end-to-end CoSeg.
[gpu_id] specifies the gpu used for training. For example, if you want to use GPU 0,1, you can specify "--gpu='0,1'".
[window_size] is the input window size. [mask_size] specifies the masked size to be reconstructed. 
[loss_coefficient] controls the balance of training : Contrastive Temporal Feature Embedding (CTFE)  and Frame Feature Reconstruction (FFR) module.
[tensorboard_flag] controls to open tensorboard to log or not.

You can specify "--resume=1 -M=[resume_model_path]" to resume training if needed.

Example command (50 Salads):
```
python3 main.py --mode=1 -F="50salads" --dataset="S" \
 --epochs 30 --batch_size 32 --lr 0.002 \
  --log_path "train_log" --window_length 10 \
 --mask_length 1 --beta 1 --tensorboard 1
```

### 3 Inference of CoSeg
```
python3 main.py --mode=2 -F=[frame_dataset_dir] --dataset=[dataset_type] \
 --batch_size [batch_size] --gpu [gpu_id]  --window_length [window_size] \
 --mask_length [mask_size] --resume=1 -M [pre-trained_model_path] \
 --smooth_factor [smooth_factor] --range [range_for_boundary_detection]
```
[frame_dataset_dir] is the specified directory to save the generated frame, which is generated in the previous step.
[dataset_type]  specifies the dataset name, "B" indicates Breakfast dataset, "S" indicates the 50 Salad dataset, "I" indicates the INRIA dataset.
[batch_size] is the training batch size. 
[gpu_id] specifies the gpu used for training. For example, if you want to use GPU 0,1, you can specify "--gpu='0,1'".
[window_size] is the input window size. [mask_size] specifies the masked size to be reconstructed. 
[pre-trained_model_path] is the model path of CoSeg model pre-trained in the previous step.
[smooth_factor] is the smooth factor for boundary detection module.
[range_for_boundary_detection] specifies the range to be considered for boundary detection module.





## Citation:
[CoSeg: Cognitively Inspired Unsupervised Generic Event Segmentation](https://arxiv.org/pdf/2109.15170.pdf).  
```
@article{wang2022coseg,
  title={CoSeg: Cognitively Inspired Unsupervised Generic Event Segmentation},
  author={Wang, Xiao and Liu, Jingen and Mei, Tao and Luo, Jiebo},
  journal={IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS (minor revision)},
  year={2022}
}
```
