# pose.py
import os
import sys
import numpy as np
# os.environ['PYOPENGL_PLATFORM'] = 'glx'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import genesis as gs
import cv2
from desk4 import create_scene
from utils.yolo_utils import detect_fruits
from utils.sam2_utils import segment_with_sam2
from utils.coordinate import annotate_and_get_3d_coords
from utils.save_json import save_json_from_detection

ENABLE_GUI = False  # 静默模式
ENABLE_GUI = True  # 渲染模式

# 路径配置
path_base = "imgs/sim_fruit_from_camera"
img_path = f"{path_base}.png"
npz_path = f"{path_base}.npz"
json_path = f"{path_base}.json"
yolo_path = "yolo_train/train_data_split/runs/detect/train/weights/best.pt"
sam_model_path = "checkpoints/sam2_b.pt"

# ✅ Step 0：仿真图像采集
def simulate_and_capture_scene(enable_gui=ENABLE_GUI):
    if enable_gui: 
        print("🎮 初始化 Genesis 场景并采集 RGB/点云")
        print("🖥️ enable_gui =", enable_gui)
        scene, xarm7, fruits, bins, camera = create_scene(enable_gui=ENABLE_GUI)
        dofs_idx = list(range(7))
        pos_init = np.array([0, 0, 0, 0.9, 0, 0.8, 0])

        for _ in range(300):
            camera.move_to_attach()
            scene.step()

        for _ in range(200):
            xarm7.set_dofs_position(pos_init, dofs_idx)
            scene.step()

        if enable_gui:
            rgb_img, _, _, _ = camera.render(rgb=True)
            rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, rgb_img_bgr)
            print(f"✅ 图像保存成功：{img_path}")

            pointcloud, mask_idx = camera.render_pointcloud(world_frame=True)[:2]
            mask = np.zeros(pointcloud.shape[:2], dtype=bool)
            mask[mask_idx] = True
            np.savez(npz_path, pointcloud=pointcloud, mask=mask)
            print(f"✅ 点云和掩码已保存为: {npz_path}")
    else:
        print("🚫 GUI 模式关闭，跳过相机渲染与点云保存")
        return None, None


    return scene, camera


# ✅ 主流程
if __name__ == "__main__":
    # Step 0: 采集图像
    scene, camera = simulate_and_capture_scene(enable_gui=ENABLE_GUI)

    # Step 1: YOLO 检测
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_path)

    # Step 2: SAM2 分割
    if len(boxes) != 0:
        masks, sam_centers, sm2_output_path = segment_with_sam2(
            image_rgb=image_rgb,
            boxes=boxes,
            sam_model_path=sam_model_path,
            input_image_path=img_path
        )
    else:
        print("❌ 未检测到任何目标，跳过后续处理")
        exit()

    # Step 3: 计算 3D 坐标
    centers = sam_centers  # or yolo_centers
    coords_3d = annotate_and_get_3d_coords(img_path, npz_path, centers)

    # Step 4: 保存 JSON
    save_json_from_detection(
        yolo_path=yolo_path,
        class_ids=class_ids,
        centers=centers,
        coords_3d=coords_3d,
        output_path=json_path
    )

    print(f"✅ 流程全部完成，结果保存在: {json_path}")

    # Step 5: 可视化（仅 GUI 模式下）
    if ENABLE_GUI and scene is not None:
        print("👁️ 保持 Genesis 可视化窗口开启，可按 Ctrl+C 手动退出")
        while True:
            scene.step()
            camera.render()
