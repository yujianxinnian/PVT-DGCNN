## VoxT-DGCNN：A 3D Object Detection Approach from Point Cloud based on Point-Voxel Transformer and Dynamic Graph CNN

**Authors**: [Qiangwen Zheng](https://github.com/yujianxinnian), [Sheng Wu*](http://adcfj.cn/sirc/door/team/TeacherList/Detail?personId=%20422), Jinghui Wei.

**Institution**: 
    1.The College of Computer and Data Science, Fuzhou University, China
    2.The Academy of Digital China (Fujian), Fuzhou University, China

## Introduction
<img src="diagram.jpg" alt="drawing" width="900" height="400"/>

### Highlights：
- Local-Global Feature Co-Learning Mechanism: Proposes VoxT-DGCNN, integrating Point-Voxel Transformer (PVFormer) and Dynamic Graph CNN (DGcnnFFN), to balance localized receptive fields with adaptive global contexts, addressing multi-scale detection challenges in autonomous driving.
- Superior Small-Object Detection: Compared with methods that only rely on point clouds and cover at least two types of detection, strong competitive performance has been achieved on KITTI and Waymo open datasets (WOD), especially in small objects, and validated through hierarchical fusion of geometric details and global semantics.
- Scenario-Specific Optimization: Establishes optimal voxel-K configurations (e.g., V=0.18/K=7 for multi-class scenes) and two-tier deployment strategies: universal model for resource-constrained environments and cascaded specialized detectors for high-compute scenarios.
- Architectural Synergy Validation: Ablation studies confirm critical contributions of DGcnnFFN, showing 10.76% accuracy drop for small objects when disabled, and full-component removal causing 18.97% AP degradation for pedestrians on WOD.

### 1. Recommended Environment
- OpenPCDet Version: 0.5.2
- Linux (tested on Ubuntu 22.04)
- Python 3.7
- PyTorch 1.9 or higher (tested on PyTorch 1.13.0)
- CUDA 9.0 or higher (tested on CUDA 11.7)


### 2. Set the Environment

```shell
pip install -r requirements.txt
python setup.py build_ext --inplace 
```



### 3. Data Preparation

- Prepare [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing)

```shell
# Download KITTI and organize it into the following form:
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2

# Generatedata infos:
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### 4. Pretrain model

PLEASE NOTE: For the voxel-based methods, the point clouds are randomly sampled, which results in some deviation in the prediction outcomes for each instance. However, the deviation is not expected to be too large. This is a normal phenomenon.


The performance (Best combination，using 11 recall poisitions) on KITTI validation set is as follows(single-stage):
```
		
Car  AP@0.70, 0.70, 0.70:

3d   AP: 89.00 78.63 77.34

Pedestrian AP@0.50, 0.50, 0.50:

3d   AP: 66.19 59.24 54.37

Cyclist AP@0.50, 0.50, 0.50:

3d   AP: 86.58 69.75 65.92
```

The performance (Best combination，using 40 recall poisitions) on the KITTI test set (two-stage).
In two-stage models are not suitable to directly report results on KITTI test set, please use slightly lower score threshold and train the models on all or 90% training data to achieve a desirable performance on KITTI test set.
```
Car  AP@0.70, 0.70, 0.70:

3D   AP: 90.43 81.61 76.97	
	
Pedestrian AP@0.50, 0.50, 0.50:

3D   AP: 51.94 44.92 41.73

Cyclist AP@0.50, 0.50, 0.50:

3D   AP: 82.57 68.58 61.69
```
Due to the voxel based method, each sampling point is random, so the results may vary during each training or testing.

### 5. Train

- Train with a single GPU

```shell

cd VoxT-DGCNN/tools
python train.py --cfg_file cfgs/kitti_models/voxt_dgcnn.yaml


```

### 6. Test with a pretrained model

```shell
cd VoxT-DGCNN/tools
python test.py --cfg_file --cfg_file ./cfgs/kitti_models/voxt_dgcnn.yaml --ckpt ${CKPT_FILE}
```
### 7. others
- To address the ultra-scale characteristics of Waymo1.2.0(https://waymo.com/open/) - whose data volume exceeds KITTI by over 20× with per-frame point cloud spatial coverage approximately 6× larger - we implemented optimized trade-offs in experimental design under computational resource constraints. Specifically, we strictly adhered to OpenPCDet(https://github.com/open-mmlab/OpenPCDet) framework conventions by utilizing approximately 20% of all training samples. For performance evaluation, comprehensive testing on the official WOD validation set was conducted, with rigorous computation of both AP and APH following the dual-difficulty-level (L1/L2) evaluation protocol, ensuring the authority and comparability of experimental results.
- "tools\cfgs\waymo_models\voxt_sgcnn.yaml", "voxt_sgcnn_two_stage_car&cyclist.yaml", and "voxt_ecovgnn_two_stage_pedestrian.yaml" uses static graph convolution (with less latency) for training

### 7. Acknowledgement
- This project is built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 
- Some codes are from [PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.DynamicEdgeConv.html#torch_geometric.nn.conv.DynamicEdgeConv) and [Voxel Set Transformer](https://github.com/skyhehe123/VoxSeT).


