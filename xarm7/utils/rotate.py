import numpy as np
import cv2
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np

def grasp_angle_to_quaternion(angle_deg):
    """
    将绕 Z 轴旋转的角度（度）转换为四元数，假设抓取方向竖直向下。
    
    参数:
        angle_deg (float): 夹爪在 XY 平面内的旋转角度，单位：度（与主轴垂直）

    返回:
        quat (np.ndarray): 长度为 4 的四元数 (x, y, z, w)
    """
    # 只绕 Z 轴旋转
    r = R.from_euler('z', angle_deg, degrees=True)
    quat = r.as_quat()  # 返回 [x, y, z, w]
    return quat

def infer_grasp_angle_and_width(mask, draw_on=None, object_index=0, pixel_to_meter=None):
    """
    计算物体主轴方向（PCA）用于夹爪朝向，并获取副轴方向上的宽度。

    参数:
        mask (np.ndarray): 单个物体的二值掩膜图像 (H, W)，uint8
        draw_on (np.ndarray): 可选，绘制箭头的背景图像（RGB）
        object_index (int): 当前物体编号，用于标注
        pixel_to_meter (float): 可选，用于将像素宽度转换为真实长度

    返回:
        grasp_angle (float): 抓取时绕Z轴旋转的角度（单位：度）
        width (float): 物体在副轴方向上的宽度（像素或米）
        output_img (np.ndarray): 绘制主轴箭头后的图像（RGB）
    """
    ys, xs = np.nonzero(mask)
    coords = np.stack((xs, ys), axis=1)

    if len(coords) < 10:
        print("⚠️ mask 太小，跳过")
        return 0, 0, draw_on if draw_on is not None else np.zeros((*mask.shape, 3), dtype=np.uint8)

    # 1. 中心化坐标
    mean = np.mean(coords, axis=0)
    centered = coords - mean

    # 2. 计算协方差矩阵并进行 PCA
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    major_axis = eigvecs[:, np.argmax(eigvals)]  # 主轴（最长方向）
    minor_axis = eigvecs[:, np.argmin(eigvals)]  # 副轴（宽度方向）

    # 3. 计算夹爪旋转角度（绕 Z 轴），令其垂直于主轴
    angle_rad = np.arctan2(major_axis[1], major_axis[0])
    angle_deg = np.degrees(angle_rad)
    grasp_angle = (angle_deg + 90) % 180

    # 4. 计算副轴方向上的投影长度（宽度）
    projections = centered @ minor_axis
    width_pixels = projections.max() - projections.min()
    width = width_pixels * pixel_to_meter if pixel_to_meter else width_pixels

    # 5. 绘图准备
    output_img = draw_on.copy() if draw_on is not None else cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    center_int = tuple(np.round(mean).astype(int))

    for vec, color, label in zip([major_axis, minor_axis], [(0, 255, 0), (0, 0, 255)], ["major", "minor"]):
        tip = (mean + vec * 50).astype(int)
        cv2.arrowedLine(output_img, center_int, tuple(tip), color=color, thickness=2, tipLength=0.2)
        cv2.putText(output_img, f"{label}_{object_index}", tuple(tip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return grasp_angle, width, output_img


def extract_grasp_infos(image_rgb, masks, pixel_to_meter=None,
                                  output_path="imgs/grasp_infos.png"):
    """
    提取每个 SAM2 mask 的抓取信息（中心点、主轴角度、副轴宽度），不依赖类别信息。

    参数:
        image_rgb (np.ndarray): 原始 RGB 图像
        masks (List[np.ndarray]): 每个物体的 mask（二维 uint8 图）
        pixel_to_meter (float): 每像素代表的真实长度（单位：米），可选
        output_path (str): 输出可视化图像保存路径

    返回:
        grasp_infos (List[dict]): 每个物体的抓取信息（中心点、角度、宽度）
        output_path (str): 保存图像的路径
    """
    draw_img = image_rgb.copy()
    grasp_infos = []

    for i, mask in enumerate(masks):
        angle, width, draw_img = infer_grasp_angle_and_width(
            mask=mask,
            draw_on=draw_img,
            object_index=i,
            pixel_to_meter=pixel_to_meter
        )

        # 计算 mask 的中心点（从 mask 本身的像素坐标平均）
        cy, cx = np.mean(np.argwhere(mask), axis=0).astype(int)

        grasp_infos.append({
            "index": i,
            "center": (cx, cy),
            "angle_deg": round(angle, 2),
            "width": round(width, 2)
        })

        # 可视化中心点和编号
        cv2.circle(draw_img, (cx, cy), radius=4, color=(255, 0, 0), thickness=-1)
        cv2.putText(draw_img, f"#{i}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

    # 保存最终图像
    draw_img_bgr = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, draw_img_bgr)

    return grasp_infos, output_path
