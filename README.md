# Our paper is accepted by Computer And Modernization！！！！

Abstract:To improve the accuracy and speed of human pose estimation, a channel-split-based human pose estimation algorithm, named Channel-Split Residual Steps Network (Channel-Split RSN), is proposed. First of all, Channel-Split Blocks are proposed to apply convolution operation for split feature in order to obtain rich feature representation. Then, Feature Enhancement Block are introduced to further split feature channel and employ different strategies for different groups which can reduce similar features in feature channels. Finally, to further enhance the Pose Refine Machine in Channel-Split RSN, combined with improved spatial attention mechanism, a Pose Refine Machine based on feature spatial correlation, named Context-PRM, is proposed. Experimental results show that on the COCO test-dev dataset, our algorithm reaches 75.9% AP and 55.36 FPS, and the Params(M) of the model is only 18.3. Compared with the traditional RSN18 and RSN50, the AP of the model is improved by 5% and 3.4%, respectively. FPS is 12.08 faster than the traditional RSN50. On the more challenging CrowdPose dataset, our approach achieves 66.9% AP and 19.16 FPS, an AP improvement of 4.6% compared to RSN18,which effectively improves the accuracy of human pose estimation and the model has a faster recognition speed.


![image](https://user-images.githubusercontent.com/68497831/128496986-3ccb60d9-fec4-4424-b485-d9cac5c72b7b.png)
![image](https://user-images.githubusercontent.com/68497831/128497007-1cbf6d46-3e14-4f13-8033-e817563f26d2.png)
![image](https://user-images.githubusercontent.com/68497831/128497024-2bef1c01-f14b-4301-a5f6-f35463cad032.png)


### if you are interested in our paper, you can download our paper in [BaiduYun](https://pan.baidu.com/s/1aoSmS6qdR5lPDQc9TtyHsA) ,password of the Baidu Yun:wmml
### You can also download our paper in [Google Drive](https://drive.google.com/file/d/194-3paObTnOe2X72RSdehgb3EztZuuIA/view?usp=sharing)


## 这里我们提供Pytorch 风格的代码(COCO 数据集)
## Here we provide Pytorch style code for future research(COCO dataset)


#### Note
* \+ means using ensemble models.
* All models are trained on 4 V100 GPUs


## Repo Structure
This repo is organized as following:
```
$Your_HOME
|-- cvpack
|
|-- dataset
|   |-- COCO
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |       |-- train2014
|   |       |-- val2014
|   |
|   
|-- lib
|   |-- models
|   |-- utils
|
|-- exps
|   |-- exp1
|   |-- exp2
|   |-- ...
|
|-- model_logs
|
|-- README.md
|-- requirements.txt
```

## Quick Start

### Installation

1. Install Pytorch referring to [Pytorch website][2].

2. Clone this repo, and config **Your Home** in **/etc/profile** or **~/.bashrc**, e.g.
 ```
 export Your_HOME='/path/of/your/cloned/repo'
 export PYTHONPATH=$PYTHONPATH:$Your_HOME
 ```

3. Install requirements:
 ```
 pip3 install -r requirements.txt
 ```

4. Install COCOAPI referring to [cocoapi website][3], or:
 ```
 git clone https://github.com/cocodataset/cocoapi.git $Your_HOME/lib/COCOAPI
 cd $Your_HOME/lib/COCOAPI/PythonAPI
 make install
 ```
 
### Dataset

#### COCO

1. Download images from [COCO website][4], and put train2014/val2014 splits into **$Your Home/dataset/COCO/images/** respectively.

2. Download ground truth from [Google Drive][6] or [Baidu Drive][10] (code: fc51), and put it into **$Your Home/dataset/COCO/gt_json/**.

3. Download detection result from [Google Drive][6] or [Baidu Drive][10] (code: fc51), and put it into **$Your Home/dataset/COCO/det_json/**.



### Log
Create a directory to save logs and models:
```
mkdir $RSN_HOME/model_logs
```

### Train
Go to specified experiment repository, e.g.
```
cd $RSN_HOME/exps/RSN18.coco
```
and run:
```
python config.py -log
python -m torch.distributed.launch --nproc_per_node=gpu_num train.py
```
the ***gpu_num*** is the number of gpus.

### Test
```
python -m torch.distributed.launch --nproc_per_node=gpu_num test.py -i iter_num
```
the ***gpu_num*** is the number of gpus, and ***iter_num*** is the iteration number you want to test.

## Citation

```
And the [code][7] of [Cascaded Pyramid Network][8] is also available. 

## Contact
You can contact us by email published in our [paper][11].

[1]: http://cocodataset.org/#keypoints-leaderboard
[2]: https://pytorch.org/
[3]: https://github.com/cocodataset/cocoapi
[4]: http://cocodataset.org/#download
[5]: http://human-pose.mpi-inf.mpg.de/
[6]: https://drive.google.com/open?id=14zW0YZ0A9kPMNt_wjBpQZg5xBiW5ecPd
[7]: https://github.com/megvii-detection/tf-cpn
[8]: https://arxiv.org/abs/1711.07319
[9]: https://github.com/fenglinglwb/MSPN
[10]: https://pan.baidu.com/s/1MqpmR7EkZu3G_Hi0_4NFTA
[11]: https://arxiv.org/abs/2003.04030
