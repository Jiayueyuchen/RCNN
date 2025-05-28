import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.structures import InstanceData
from matplotlib import pyplot as plt
from mmcv.ops import nms
from mmengine.visualization import LocalVisBackend


def get_rpn_proposals(model, img_path):
    """安全提取RPN候选框（颜色参数已修复）"""
    device = next(model.parameters()).device

    try:
        img = mmcv.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{img_path}")

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
        img_tensor = img_tensor.unsqueeze(0)

        data_sample = InstanceData()
        data_sample.set_metainfo({
            'img_shape': img_tensor.shape[2:],
            'ori_shape': img.shape[:2],
            'scale_factor': (1.0, 1.0)
        })

        with torch.no_grad():
            feats = model.extract_feat(img_tensor)
            rpn_results = model.rpn_head.predict(feats, [data_sample])

        proposals = rpn_results[0].bboxes.cpu().numpy().astype(np.float32)
        scores = rpn_results[0].scores.cpu().numpy().astype(np.float32)

        return proposals, scores

    except Exception as e:
        print(f"提取RPN候选框时发生错误：{str(e)}")
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)


def visualize_results(img_path, model, proposals, scores, out_dir='results'):
    """可视化对比（颜色参数已修复）"""
    try:
        os.makedirs(out_dir, exist_ok=True)

        # ========== 初始化可视化工具 ==========
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

        # 关键修复：正确初始化可视化后端
        if 'local' not in visualizer._vis_backends:
            visualizer._vis_backends['local'] = LocalVisBackend(save_dir=out_dir)
        else:
            visualizer._vis_backends['local'].save_dir = out_dir

        # ========== 处理RPN候选框 ==========
        if len(proposals) == 0 or len(scores) == 0:
            print(f"警告：{os.path.basename(img_path)} 未检测到候选框")
            return

        # 关键修改1：增强分数过滤
        score_threshold = 0.7  # 调高阈值
        valid_mask = scores >= score_threshold
        proposals = proposals[valid_mask]
        scores = scores[valid_mask]

        # 关键修改2：更严格的NMS
        if len(proposals) > 1:
            proposals_tensor = torch.from_numpy(proposals).float()
            scores_tensor = torch.from_numpy(scores).float()
            keep_idx = nms(proposals_tensor, scores_tensor, iou_threshold=0.5)  # 降低IoU阈值

            if isinstance(keep_idx, tuple):
                keep_idx = keep_idx[0]
            keep_idx = keep_idx.cpu().numpy().astype(int)
            keep_idx = keep_idx[keep_idx < len(proposals)]
            proposals = proposals[keep_idx]
            scores = scores[keep_idx]

        # 关键修改3：限制显示数量
        max_show = 100  # 根据图像尺寸调整
        if len(proposals) > max_show:
            proposals = proposals[np.argsort(scores)[-max_show:]]

        # ========== 绘制RPN候选框 ==========
        img = mmcv.imread(img_path)
        h, w = img.shape[:2]

        # 坐标裁剪
        proposals[:, 0::2] = np.clip(proposals[:, 0::2], 0, w)
        proposals[:, 1::2] = np.clip(proposals[:, 1::2], 0, h)

        visualizer.set_image(img)
        visualizer.draw_bboxes(
            proposals,
            edge_colors=(0.0, 1.0, 0.0),  # 修正为归一化RGB值 (R,G,B)
            line_widths=1,
            line_styles='--',
            alpha=0.3
        )
        proposal_img = visualizer.get_image()

        # ========== 处理最终检测 ==========
        result = inference_detector(model, img)
        final_result = result.pred_instances

        # 结果过滤
        keep = (final_result.scores >= 0.3) & \
               (final_result.bboxes[:, 0] >= 0) & \
               (final_result.bboxes[:, 1] >= 0) & \
               (final_result.bboxes[:, 2] <= w) & \
               (final_result.bboxes[:, 3] <= h)
        final_result = final_result[keep]

        bboxes = final_result.bboxes.cpu().numpy()
        labels = final_result.labels.cpu().numpy()
        scores_final = final_result.scores.cpu().numpy()

        # ========== 绘制最终结果 ==========
        visualizer.set_image(img)

        # 绘制掩码
        if 'masks' in final_result and len(final_result.masks) > 0:
            masks = final_result.masks.cpu().numpy()
            visualizer.draw_binary_masks(
                masks,
                colors=(1.0, 0.0, 0.0),  # 归一化RGB红色
                alphas=0.3
            )

        # 绘制边界框
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
        visualizer.draw_bboxes(
            bboxes,
            edge_colors=(1.0, 0.0, 0.0),  # 归一化RGB红色
            line_widths=2
        )

        # 绘制标签
        for i in range(len(bboxes)):
            x1, y1 = int(bboxes[i][0]), int(bboxes[i][1])
            if x1 < 0 or y1 < 0 or x1 >= w or y1 >= h:
                continue

            positions = np.array([[x1, max(10, y1 - 10)]], dtype=np.int32)
            # label = model.dataset_meta['classes'][labels[i]]
            label = 'sheep'
            visualizer.draw_texts(
                texts=[f'{label} {scores_final[i]:.2f}'],
                positions=positions,
                font_sizes=10,
                colors=(1.0, 1.0, 1.0),  # 白色（归一化）
                vertical_alignments='bottom',
                bboxes=dict(facecolor=(1.0, 0.0, 0.0, 0.5))  # 红色背景（RGBA归一化值）
            )

        final_img = visualizer.get_image()

        # ========== 生成对比图 ==========
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        axes[0].imshow(cv2.cvtColor(proposal_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'RPN Proposals', fontsize=14)
        axes[1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Final Detections', fontsize=14)

        for ax in axes:
            ax.axis('off')

        output_path = os.path.join(out_dir, os.path.basename(img_path))
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

    except Exception as e:
        print(f"处理图像 {img_path} 时发生错误：{repr(e)}")


if __name__ == '__main__':
    # 配置（保持原始路径）
    config_path = '../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    checkpoint_path = '../work_dirs/mask-rcnn/epoch_9.pth'
    test_images = [
        '../data/COCO/val2017/2007_000027.jpg',
        '../data/COCO/val2017/2007_000063.jpg',
        '../data/COCO/val2017/2007_000256.jpg'
        # '../data/COCO/val2017/2007_000392.jpg',
        # '../data/COCO/val2017/2007_000480.jpg',
        # '../data/COCO/val2017/2007_000559.jpg',
        # '../data/COCO/val2017/2007_000584.jpg',
        # '../data/COCO/val2017/2007_000661.jpg',
        # '../data/COCO/val2017/2007_000713.jpg',
        # '../data/COCO/val2017/2007_000727.jpg',
        # '../data/COCO/val2017/2007_000733.jpg',
        # '../data/COCO/val2017/2007_000904.jpg',
        # '../data/COCO/val2017/2007_000999.jpg',
        # '../data/COCO/val2017/2007_001027.jpg',
        # '../data/COCO/val2017/2007_001185.jpg',
        # '../data/COCO/val2017/2007_001321.jpg',
        # '../data/COCO/val2017/2007_001397.jpg',
        # '../data/COCO/val2017/2007_001487.jpg',
        # '../data/COCO/val2017/2007_001602.jpg',
        # '../data/COCO/val2017/2007_001667.jpg',
        # '../data/COCO/val2017/2007_001678.jpg',
        # '../data/COCO/val2017/2007_001733.jpg'
    ]
    output_dir = '../results/visualization'

    # 初始化模型
    try:
        model = init_detector(config_path, checkpoint_path, device='cuda:0')
    except Exception as e:
        print(f"模型初始化失败：{repr(e)}")
        exit(1)

    # 处理图像
    success_count = 0
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"文件不存在：{img_path}")
            continue

        try:
            proposals, scores = get_rpn_proposals(model, img_path)
            visualize_results(img_path, model, proposals, scores, output_dir)
            success_count += 1
        except Exception as e:
            print(f"处理 {os.path.basename(img_path)} 失败：{repr(e)}")

    print(f"\n处理结果：成功 {success_count}/{len(test_images)}")
    print(f"输出目录：{os.path.abspath(output_dir)}")