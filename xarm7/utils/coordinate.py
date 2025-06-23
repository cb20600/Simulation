# import cv2
# import numpy as np

# def annotate_and_get_3d_coords(image_path, npz_path, pixel_points):
#     """
#     给定图像和3D点云数据，在图像上标注像素点并打印其3D坐标。

#     参数:
#         image_path (str): 图像路径 (.png)
#         npz_path (str): 包含 pointcloud 和 mask 的 .npz 文件路径
#         pixel_points (list of tuple): [(u, v), ...] 格式的像素坐标
#     """

#     # 加载图像与数据
#     image = cv2.imread(image_path)
#     data = np.load(npz_path)
#     pointcloud = data['pointcloud']
#     mask = data['mask']

#     annotated_image = image.copy()

#     for (u, v) in pixel_points:
#         if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
#             if mask[v, u]:
#                 coord_3d = pointcloud[v, u]
#                 print(f"✅ 像素点 ({u}, {v}) 的 3D 坐标为: {coord_3d}")
#                 cv2.circle(annotated_image, (u, v), 6, (0, 0, 255), -1)
#                 cv2.putText(annotated_image, f"{u},{v}", (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             else:
#                 print(f"⚠️ 像素点 ({u}, {v}) 的深度无效")
#         else:
#             print(f"❌ 像素点 ({u}, {v}) 超出图像范围")
    
#     output_path = image_path.replace(".png", "_annotated.png")
#     cv2.imwrite(output_path, annotated_image)
#     print(f"✅ 可视化图像已保存为 {output_path}")





import numpy as np
import cv2


def annotate_and_get_3d_coords(image_path, npz_path, pixel_points):
    """
    从像素点列表中提取 3D 坐标，并在图像上可视化。

    参数:
        image_path (str): 图像路径
        npz_path (str): 保存的 .npz 文件路径（包含 pointcloud 和 mask）
        pixel_points (list of (u, v)): 图像中心点坐标

    返回:
        coords_3d (list of [x, y, z] or None): 每个像素点对应的世界坐标
    """
    image = cv2.imread(image_path)
    data = np.load(npz_path)
    pointcloud = data['pointcloud']
    mask = data['mask']

    annotated_image = image.copy()
    coord_list = []

    for (u, v) in pixel_points:
        if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
            if mask[v, u]:
                coord_3d = pointcloud[v, u]
                coord_list.append(coord_3d.tolist())
                cv2.circle(annotated_image, (u, v), 6, (0, 0, 255), -1)
                cv2.putText(annotated_image, f"{u},{v}", (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                coord_list.append(None)
                print(f"⚠️ 无效深度: ({u}, {v})")
        else:
            coord_list.append(None)
            print(f"❌ 超出图像范围: ({u}, {v})")

    output_path = image_path.replace(".png", "_annotated.png")
    cv2.imwrite(output_path, annotated_image)
    print(f"✅ 可视化图像已保存为: {output_path}")

    return coord_list



if __name__ == "__main__":
    # 示例参数
    image_path = "test.png"
    npz_path = "camera_data.npz"
    pixel_points = [(500, 600), (450, 600), (100, 200)]

    annotate_and_get_3d_coords(image_path, npz_path, pixel_points)
