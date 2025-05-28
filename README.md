首先通过git clone或者直接下载并解压到本地，在终端进入文件所在地址。

之后可以在文件中新建一个data文件夹，给VOC数据集下载到data文件夹中，运行data文件夹里的voc2coco.py文件可以生成一个COCO文件夹，将VOC数据转成COCO数据集格式。然后运行mask_err.py，将生成的数据集替换原始数据集里的instances_train2017.json和instances_val2017.json

训练maskrcnn模型在终端用 set KMP_DUPLICATE_LIB_OK=TRUE && python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/mask-rcnn 命令，训练sparsercnn模型在终端用 set KMP_DUPLICATE_LIB_OK=TRUE && python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py --work-dir work_dirs/sparse-rcnn 命令

然后可以进入HW文件夹，通过 Python RPN_proposal_boxes.py 输出proposal box和最终结果对比图，通过 Python compare_models.py 输出两个模型结果对比，通过 python cross_dataset_test.py 输出模型对自定义的三张图的检验结果
