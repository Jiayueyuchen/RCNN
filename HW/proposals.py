import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 修复 OpenMP 冲突
import cv2
import numpy as np
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures.bbox import nms
from mmengine.structures import InstanceData
from matplotlib import pyplot as plt


def get_rpn_proposals(model, img_path):
    """提取 RPN 生成的候选框（优化版本）"""
    device = next(model.parameters()).device
    img = mmcv.imread(img_path)

    # 构建数据样本
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
    data_sample = InstanceData()
    data_sample.set_metainfo({
        'img_shape': img_tensor.shape[1:],
        'ori_shape': img.shape[:2],
        'scale_factor': (1.0, 1.0)
    })

    # 前向传播
    feats = model.extract_feat(img_tensor.unsqueeze(0))
    rpn_results = model.rpn_head.predict(feats, [data_sample])

    # 转换为 numpy
    proposals = rpn_results[0].bboxes.cpu().numpy()
    scores = rpn_results[0].scores.cpu().numpy()
    return proposals, scores


def visualize_results(img_path, model, proposals, scores, out_dir='results'):
    """可视化对比（优化版本）"""
    os.makedirs(out_dir, exist_ok=True)

    # ================ RPN Proposals 优化处理 ================
    # 1. 分数阈值筛选
    score_threshold = 0.5
    valid_mask = scores >= score_threshold
    proposals = proposals[valid_mask]
    scores = scores[valid_mask]

    # 2. NMS 处理
    if len(proposals) > 0:
        proposals_tensor = torch.from_numpy(proposals).float()
        scores_tensor = torch.from_numpy(scores).float()
        keep_idx = nms(proposals_tensor, scores_tensor, iou_threshold=0.7)
        proposals = proposals[keep_idx.numpy().astype(int)]
        scores = scores[keep_idx.numpy().astype(int)]

    # 3. 数量限制
    max_proposals = 200
    if len(proposals) > max_proposals:
        top_k = np.argsort(scores)[-max_proposals:]
        proposals = proposals[top_k]
        scores = scores[top_k]

    # ================ 可视化初始化 ================
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    img = mmcv.imread(img_path)
    h, w = img.shape[:2]

    # ================ 绘制 RPN Proposals ================
    visualizer.set_image(img)
    visualizer.draw_bboxes(
        proposals,
        edge_colors=(0.2, 0.8, 0.2),  # 浅绿色
        line_widths=1,
        alpha=0.3,
        line_styles=':'
    )
    proposal_img = visualizer.get_image()

    # ================ 绘制最终结果 ================
    result = inference_detector(model, img)
    final_result = result.pred_instances

    # 结果过滤
    keep = (final_result.scores >= 0.5) & \
           (final_result.bboxes[:, 0] >= 0) & \
           (final_result.bboxes[:, 1] >= 0) & \
           (final_result.bboxes[:, 2] <= w) & \
           (final_result.bboxes[:, 3] <= h)
    final_result = final_result[keep]

    # 转换为 numpy
    bboxes = final_result.bboxes.cpu().numpy()
    labels = final_result.labels.cpu().numpy()
    scores = final_result.scores.cpu().numpy()

    # 绘制最终检测
    visualizer.set_image(img)
    if 'masks' in final_result:
        visualizer.draw_binary_masks(
            final_result.masks.cpu().numpy(),
            colors='pink',
            alphas=0.3
        )
    visualizer.draw_bboxes(
        bboxes,
        edge_colors=(1.0, 0.0, 0.0),
        line_widths=2
    )

    # 绘制标签（可选）
    for i in range(len(bboxes)):
        label = model.dataset_meta['classes'][labels[i]]
        x1, y1, _, _ = bboxes[i].astype(int)
        visualizer.draw_texts(
            texts=[f'{label} {scores[i]:.2f}'],
            positions=np.array([[x1, y1 - 10]], dtype=np.int32),
            font_sizes=10,
            colors=(1.0, 1.0, 1.0),
            vertical_alignments='bottom'
        )
    final_img = visualizer.get_image()

    # ================ 生成对比图 ================
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(cv2.cvtColor(proposal_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'RPN Proposals ({len(proposals)})')
    axes[1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Final Detections ({len(bboxes)})')

    output_path = os.path.join(out_dir, os.path.basename(img_path))
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == '__main__':
    config_path = '../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    checkpoint_path = '../work_dirs/mask-rcnn/epoch_9.pth'
    test_images = [  # 替换为你的测试图片路径
        '../data/COCO/val2017/2007_000027.jpg',
        '../data/COCO/val2017/2007_000063.jpg',
        '../data/COCO/val2017/2007_000256.jpg'
    ]
    output_dir = '../results/visualization'

    model = init_detector(config_path, checkpoint_path, device='cuda:0')

    for img_path in test_images:
        if os.path.exists(img_path):
            proposals, scores = get_rpn_proposals(model, img_path)
            visualize_results(img_path, model, proposals, scores, output_dir)

    print(f"结果保存至：{os.path.abspath(output_dir)}")