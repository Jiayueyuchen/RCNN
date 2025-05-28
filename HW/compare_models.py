import os
import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from matplotlib import pyplot as plt


def compare_models(img_path, mask_rcnn, sparse_rcnn, out_dir='results/model_comparison'):
    """对比两个模型在同一张图上的结果"""
    os.makedirs(out_dir, exist_ok=True)

    # 初始化可视化工具
    visualizer = VISUALIZERS.build(mask_rcnn.cfg.visualizer)
    visualizer.dataset_meta = mask_rcnn.dataset_meta

    # 读取图像
    img = mmcv.imread(img_path)

    # --------------------- Mask R-CNN 结果 ---------------------
    mask_result = inference_detector(mask_rcnn, img)
    mask_instances = mask_result.pred_instances

    # 过滤低分结果
    keep = mask_instances.scores >= 0.5
    mask_instances = mask_instances[keep]

    visualizer.set_image(img.copy())
    if 'masks' in mask_instances:
        visualizer.draw_binary_masks(
            mask_instances.masks.cpu().numpy(),
            colors='pink',
            alphas=0.3
        )
    visualizer.draw_bboxes(
        mask_instances.bboxes.cpu().numpy(),
        edge_colors=(1.0, 0.0, 0.0),  # 红色
        line_widths=2
    )
    mask_img = visualizer.get_image()

    # --------------------- Sparse R-CNN 结果 ---------------------
    sparse_result = inference_detector(sparse_rcnn, img)
    sparse_instances = sparse_result.pred_instances

    keep = sparse_instances.scores >= 0.4
    sparse_instances = sparse_instances[keep]

    visualizer.set_image(img.copy())
    if 'masks' in sparse_instances:
        visualizer.draw_binary_masks(
            sparse_instances.masks.cpu().numpy(),
            colors='cyan',
            alphas=0.3
        )
    visualizer.draw_bboxes(
        sparse_instances.bboxes.cpu().numpy(),
        edge_colors=(0.0, 0.0, 1.0),  # 蓝色
        line_widths=2
    )
    sparse_img = visualizer.get_image()

    # --------------------- 生成对比图 ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    axes[0].imshow(mmcv.bgr2rgb(mask_img))
    axes[0].set_title('Mask R-CNN (Red)')
    axes[1].imshow(mmcv.bgr2rgb(sparse_img))
    axes[1].set_title('Sparse R-CNN (Blue)')

    output_path = os.path.join(out_dir, os.path.basename(img_path))
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == '__main__':
    # --------------- 配置区 ---------------
    mask_config = '../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    mask_checkpoint = '../work_dirs/mask-rcnn/epoch_9.pth'
    sparse_config = '../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'
    sparse_checkpoint = '../work_dirs/sparse-rcnn/epoch_12.pth'
    test_images = [
        '../data/COCO/val2017/2007_000027.jpg',
        '../data/COCO/val2017/2007_000063.jpg',
        '../data/COCO/val2017/2007_000256.jpg'
    ]
    output_dir = 'results/model_comparison'
    # -------------------------------------

    # 初始化模型
    mask_rcnn = init_detector(mask_config, mask_checkpoint, device='cuda:0')
    sparse_rcnn = init_detector(sparse_config, sparse_checkpoint, device='cuda:0')

    # 处理每张测试图像
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"图像 {img_path} 不存在，跳过")
            continue
        compare_models(img_path, mask_rcnn, sparse_rcnn, output_dir)

    print(f"模型对比结果已保存至：{os.path.abspath(output_dir)}")