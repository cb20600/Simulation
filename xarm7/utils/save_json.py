import json
from ultralytics import YOLO


# def save_json_from_detection(
#     class_ids,
#     centers,
#     coords_3d,
#     output_path,
#     yolo_path=None,
#     class_names=None,
#     widths=None,
#     quaternions=None
# ):
#     """
#     通用保存函数，可选择提供 YOLO 模型路径或直接提供类别名映射表

#     参数:
#         class_ids (List[int])
#         centers (List[Tuple[int, int]])
#         coords_3d (List[Tuple[float, float, float]])
#         output_path (str)
#         yolo_path (str, optional): 提供时，用于获取类别名映射
#         class_names (Dict[int, str], optional): 自定义 ID 到名称映射
#         widths (List[float], optional): 抓取宽度
#         quaternions (List[List[float]], optional): 四元数姿态
#     """

#     # 获取类别名映射
#     if class_names is not None:
#         names = class_names
#     elif yolo_path is not None:
#         names = YOLO(yolo_path).names
#     else:
#         names = {}  # 不映射类别名，仅保留 ID

#     assert len(class_ids) == len(centers) == len(coords_3d)

#     data = []
#     for i, (cls_id, center, coord) in enumerate(zip(class_ids, centers, coords_3d)):
#         if coord is None:
#             continue

#         name = names.get(int(cls_id), f"id:{int(cls_id)}")

#         entry = {
#             "class": name,
#             "center": [int(c) for c in center],
#             "coord_3d": [round(float(x), 3) for x in coord]
#         }

#         if widths is not None:
#             entry["grasp_width"] = round(widths[i], 4)
#         if quaternions is not None:
#             entry["grasp_quaternion"] = [round(x, 4) for x in quaternions[i]]

#         data.append(entry)

#         print(f"{entry['class']} @ {entry['coord_3d']}", end="")
#         if "grasp_width" in entry:
#             print(f", width={entry['grasp_width']}", end="")
#         if "grasp_quaternion" in entry:
#             print(f", quat={entry['grasp_quaternion']}", end="")
#         print()

#     with open(output_path, "w") as f:
#         json.dump(data, f, indent=2)

#     print(f"\n✅ 保存完成：{output_path}")
import json
from ultralytics import YOLO

def save_json_from_detection(
    class_ids,
    centers,
    coords_3d,
    output_path,
    yolo_path=None,
    class_names=None,
    widths=None,
    quaternions=None
):
    """
    通用 JSON 保存函数，兼容 YOLO/SAM 自动分割

    参数:
        class_ids (List[int])
        centers (List[Tuple[int, int]])
        coords_3d (List[Tuple[float, float, float]])
        output_path (str): JSON 文件路径
        yolo_path (str, optional): 若提供，将用于获取类别名映射
        class_names (Dict[int, str], optional): 自定义类别名映射
        widths (List[float], optional): 抓取宽度（米）
        quaternions (List[List[float]], optional): 四元数姿态
    """
    # 1. 加载类别名映射
    if class_names is not None:
        names = class_names
    elif yolo_path is not None:
        names = YOLO(yolo_path).names
    else:
        names = {}

    assert len(class_ids) == len(centers) == len(coords_3d)

    data = []
    for i, (cls_id, center, coord) in enumerate(zip(class_ids, centers, coords_3d)):
        if coord is None:
            continue

        label = names.get(int(cls_id), f"id:{int(cls_id)}")

        entry = {
            "class": label,
            "center": [int(center[0]), int(center[1])],
            "coord_3d": [round(float(coord[0]), 3), round(float(coord[1]), 3), round(float(coord[2]), 3)]
        }

        if widths:
            entry["grasp_width"] = round(widths[i], 4)

        if quaternions:
            entry["grasp_quaternion"] = [round(q, 4) for q in quaternions[i]]

        data.append(entry)

        print(f"{entry['class']} @ {entry['coord_3d']}", end="")
        if "grasp_width" in entry:
            print(f", width={entry['grasp_width']}", end="")
        if "grasp_quaternion" in entry:
            print(f", quat={entry['grasp_quaternion']}", end="")
        print()

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ 保存成功：{output_path}")


if __name__ == "__main__":
    # 示例调用
    save_json_from_detection(
        yolo_path="yolo_train/train_data_split/runs/detect/train/weights/best.pt",
        class_ids=[0, 1],
        centers=[[100, 200], [150, 250]],
        coords_3d=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        output_path="output.json"
    )
