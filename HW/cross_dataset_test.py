import os
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import numpy as np
from matplotlib import pyplot as plt


def visualize_custom_result(img_path, model, out_dir='results/custom_images'):
    """在自定义图像上可视化结果"""
    os.makedirs(out_dir, exist_ok=True)

    # 初始化可视化工具
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 读取图像
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    instances = result.pred_instances

    # 过滤低分结果
    keep = instances.scores >= 0.5
    instances = instances[keep]

    visualizer.set_image(img)

    # 绘制掩码
    if 'masks' in instances:
        visualizer.draw_binary_masks(
            instances.masks.cpu().numpy(),
            colors='yellow',
            alphas=0.3
        )

    # 绘制检测框
    visualizer.draw_bboxes(
        instances.bboxes.cpu().numpy(),
        edge_colors=(0.0, 1.0, 0.0),  # 绿色
        line_widths=2
    )

    # 绘制标签
    for i in range(len(instances.bboxes)):
        label = model.dataset_meta['classes'][instances.labels[i].item()] ###
        score = instances.scores[i].item()
        x1, y1, _, _ = instances.bboxes[i].cpu().numpy().astype(int)

        visualizer.draw_texts(
        texts = [f'{label} {score:.2f}'],
        positions = np.array([[x1, y1 - 10]], dtype=np.int32),
        font_sizes = 10,
        colors = (1.0, 1.0, 1.0),
        vertical_alignments = 'bottom',
        bboxes = dict(facecolor=(0.0, 1.0, 0.0, 0.5))  # 绿色背景
    )

    output_img = visualizer.get_image()
    output_path = os.path.join(out_dir, os.path.basename(img_path))
    mmcv.imwrite(mmcv.rgb2bgr(output_img), output_path)


# 修改 cross_dataset_test.py 的主函数部分
if __name__ == '__main__':
    # 初始化两个模型
    mask_rcnn = init_detector(
        '../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
        '../work_dirs/mask-rcnn/epoch_9.pth',
        device='cuda:0'
    )
    sparse_rcnn = init_detector(
        '../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py',
        '../work_dirs/sparse-rcnn/epoch_12.pth',
        device='cuda:0'
    )

    # 处理自定义图像
    custom_images = [
        '../data/custom_images/image1.jpg',
        '../data/custom_images/image2.jpg',
        '../data/custom_images/image3.jpg'
        # '../data/custom_images/image4.jpg',
        # '../data/custom_images/image5.jpg',
        # '../data/custom_images/image6.jpg',
        # '../data/custom_images/image7.jpg',
        # '../data/custom_images/image8.jpg',
        # '../data/custom_images/image9.jpg',
        # '../data/custom_images/image10.jpg',
        # '../data/custom_images/image11.jpg',
        # '../data/custom_images/image12.jpg',
        # '../data/custom_images/image13.jpg'
    ]
    for img_path in custom_images:
        if not os.path.exists(img_path):
            print(f"图像 {img_path} 不存在，跳过")
            continue
        visualize_custom_result(img_path, mask_rcnn, 'results/custom/mask_rcnn')
        visualize_custom_result(img_path, sparse_rcnn, 'results/custom/sparse_rcnn')
    print(f"自定义图像结果已保存")
