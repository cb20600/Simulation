# -------- 第一部分：保存图像和点云数据 --------

import os
import numpy as np
import genesis as gs
from utils.desk4 import create_scene
import cv2

os.environ['PYOPENGL_PLATFORM'] = 'glx'

# 创建场景
scene, xarm7, fruits, bins, camera = create_scene()

# 设置初始关节角度
pos_init = np.array([0, 0, 0, 1, 0, 0.8, 0])
dofs_idx = list(range(7))

# # 等待场景稳定
# for _ in range(300):
#     camera.move_to_attach()
#     scene.step()

for _ in range(200):
    xarm7.set_dofs_position(pos_init, dofs_idx)
    scene.step()

# 渲染 RGB 图像并保存
rgb_img, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
save_rgb_path = "camera_rgb.png"
cv2.imwrite(save_rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
print(f"✅ 图像保存成功：{save_rgb_path}")

# 渲染点云和掩码
pointcloud, mask_idx = camera.render_pointcloud(world_frame=True)[:2]
mask = np.zeros(pointcloud.shape[:2], dtype=bool)
mask[mask_idx] = True


# 保存为 npz 格式
np.savez("camera_data.npz", pointcloud=pointcloud, mask=mask)
print("✅ 点云和掩码已保存为 camera_data.npz")


# -------- 第二部分：加载图像和数据，分析用户指定像素点的 3D 坐标并可视化 --------


import numpy as np
import cv2

# 加载图像与数据
image = cv2.imread("camera_rgb.png")
data = np.load("camera_data.npz")
pointcloud = data['pointcloud']
mask = data['mask']

# 用户指定的像素点列表（注意格式为 (v, u)，即 (row, col)）
pixel_points = [(500,600),(450, 600), (100,200)]  # 请在此处填写像素点列表，例如 [(500, 600), (300, 400)]

# 拷贝图像用于标注
annotated_image = image.copy()

# 遍历并输出对应的 3D 坐标，并在图像上标注
for (v, u) in pixel_points:
    if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
        if mask[v, u]:
            coord_3d = pointcloud[v, u]
            print(f"✅ 像素点 ({u}, {v}) 的 3D 坐标为: {coord_3d}")
            cv2.circle(annotated_image, (u, v), 6, (0, 0, 255), -1)
            cv2.putText(annotated_image, f"{u},{v}", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            print(f"⚠️ 像素点 ({u}, {v}) 的深度无效")
    else:
        print(f"❌ 像素点 ({u}, {v}) 超出图像范围")

# 保存可视化结果
cv2.imwrite("camera_annotated.png", annotated_image)
print("✅ 可视化图像已保存为 camera_annotated.png")


