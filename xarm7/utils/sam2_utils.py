# sam2_utils.py
from ultralytics import SAM
import numpy as np
import cv2
import os
from .rotate import extract_grasp_infos
from .rotate import grasp_angle_to_quaternion

def segment_with_sam2(image_rgb, boxes, sam_model_path, input_image_path):
    """
    使用 Ultralytics SAM2 对 YOLO 检测框进行精细分割，并绘制中心点

    参数:
        image_rgb (np.ndarray): RGB 图像 (H, W, 3)
        boxes (list of tuple): YOLO 检测框 [(x1, y1, x2, y2), ...]
        sam_model_path (str): SAM2 权重路径，如 "sam2_b.pt"
        input_image_path (str): 原图路径，用于保存输出图像

    返回:
        out_masks (list of np.ndarray): 每个目标的 mask (H, W)
        centers (list of tuple): 每个 mask 的中心点 (u, v)
        saved_path (str): 保存带有 mask + 中心点 的图像路径
    """
    model = SAM(sam_model_path)

    # 由于 Ultralytics 只能读文件路径，所以先保存临时图像
    temp_input_path = input_image_path
    if isinstance(image_rgb, np.ndarray):
        cv2.imwrite(temp_input_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    # 转换 box 为 list[int] 格式
    bbox_list = [list(map(int, box)) for box in boxes]

    # 推理
    results = model(temp_input_path, bboxes=bbox_list)

    # 获取原图作为底图
    output_img = results[0].plot()  # ndarray
    output_img = output_img.copy()

    # 提取 masks 和中心点
    out_masks = []
    centers = []
    for i, mask in enumerate(results[0].masks.data):
        mask_np = mask.cpu().numpy().astype(np.uint8)
        out_masks.append(mask_np)

        if np.any(mask_np):
            yx = np.argwhere(mask_np)
            cy, cx = yx.mean(axis=0).astype(int)
            centers.append((cx, cy))

            # 绘制中心点
            cv2.circle(output_img, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(output_img, f"#{i}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            centers.append((0, 0))

    # 保存带中心点的图像
    save_path = input_image_path.replace(".png", "_sam2_masks_with_centers.png")
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    print(f"✅ Ultralytics SAM2 分割完成，共 {len(centers)} 个目标")
    print(f"✅ 已绘制中心点并保存图像：{save_path}")
    return out_masks, centers, save_path


if __name__ == "__main__":
    from yolo_utils import detect_fruits

    image_path = "test.png"
    sam_model_path = "checkpoints/sam2_b.pt"
    yolo_model_path = "checkpoints/best_yolo.pt"

    # 使用你原来的 YOLO 检测函数
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(image_path, yolo_model_path)

    # 使用 Ultralytics SAM2 做精细分割
    masks, sam_centers, result_path = segment_with_sam2(image_rgb, boxes, sam_model_path, image_path)
    print("SAM2中心点：", sam_centers)

    grasp_infos, output_img_path = extract_grasp_infos(
    image_rgb=image_rgb,
    masks=masks,
    pixel_to_meter=0.0025,  # 可选：像素转换为米的比例
    output_path="grasp_infos_visual.png"
    )

    # 输出结果
    print("✅ 抓取信息如下：")
    for info in grasp_infos:
        print(info)
    for info in grasp_infos:
        angle = info["angle_deg"]
        quat = grasp_angle_to_quaternion(angle)
        print(f"物体 #{info['index']} 四元数姿态: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]")


    print("✅ 可视化图已保存至：", output_img_path)
