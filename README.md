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
 git clone https://github.com/cocodataset/cocoapi.git $RSN_HOME/lib/COCOAPI
 cd $RSN_HOME/lib/COCOAPI/PythonAPI
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
