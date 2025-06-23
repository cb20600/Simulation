import json
from ultralytics import YOLO

def save_json_from_detection(yolo_path, class_ids, centers, coords_3d, output_path):
    yolo_model = YOLO(yolo_path)
    names = yolo_model.names

    assert len(class_ids) == len(centers) == len(coords_3d)

    data = []
    for cls_id, center, coord in zip(class_ids, centers, coords_3d):
        if coord is None:
            continue
        entry = {
            "class": names[int(cls_id)] if int(cls_id) in names else f"id:{int(cls_id)}",
            "center": [int(c) for c in center],
            "coord_3d": [round(float(x), 2) for x in coord]
        }
        data.append(entry)
        x, y, z = entry["coord_3d"]
        print(f"{entry['class']:<12} @ ({x:.2f}, {y:.2f}, {z:.2f})")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ 信息已保存至 {output_path}")


if __name__ == "__main__":
    # 示例调用
    save_json_from_detection(
        yolo_path="yolo_train/train_data_split/runs/detect/train/weights/best.pt",
        class_ids=[0, 1],
        centers=[[100, 200], [150, 250]],
        coords_3d=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        output_path="output.json"
    )
