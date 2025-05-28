# 目标检测项目使用指南

## 环境准备
1. 克隆仓库或下载源码：
```bash
git clone [仓库地址]
```
# 或下载ZIP文件并解压
进入项目目录：

bash
cd [项目目录]
数据集准备
创建数据目录：

bash
mkdir data
下载VOC数据集至data文件夹

转换数据集格式：

bash
python data/voc2coco.py
更新数据集：

bash
python data/mask_err.py
模型训练
Mask R-CNN 训练
bash
set KMP_DUPLICATE_LIB_OK=TRUE && python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/mask-rcnn
Sparse R-CNN 训练
bash
set KMP_DUPLICATE_LIB_OK=TRUE && python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/sparse-rcnn
结果可视化
进入HW目录运行：

bash
cd HW
生成RPN proposal对比图：

bash
python RPN_proposal_boxes.py
模型结果对比：

bash
python compare_models.py
自定义图像测试：

bash
python cross_dataset_test.py
