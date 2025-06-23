# yolo_utils.py
from ultralytics import YOLO
import cv2

# 初始化 YOLO 模型（只加载一次）
# yolo_yolo_path = "checkpoints/best.pt"
# yolo_path = ""checkpoints/yolov8s.pt"


def detect_fruits(input_path, yolo_path):
    """
    使用 YOLO 检测图像中的果蔬

    参数:
        input_path (str): 图像路径

    返回:
        image_rgb (np.ndarray): 图像 (RGB)
        boxes (list of tuple): [(x1, y1, x2, y2), ...]
        centers (list of tuple): [(u, v), ...] 中心点坐标
        class_ids (list of int): 每个框的类别
    """
    
    # image_bgr = cv2.imread(input_path)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # results = yolo_model(image_rgb)
    # boxes = []
    # centers = []
    # class_ids = []

    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    #     cls = int(box.cls[0].item())
    #     boxes.append((int(x1), int(y1), int(x2), int(y2)))
    #     centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
    #     class_ids.append(cls)



    yolo_model = YOLO(yolo_path)
    print("✅ YOLO 模型实际输入尺寸 imgsz =", yolo_model.model.args["imgsz"])
    results = yolo_model(input_path)  # ✅ 推荐：直接用路径
    image_bgr = cv2.imread(input_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    boxes, centers, class_ids = [], [], []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].item())
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
        centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        class_ids.append(cls)

    # 加在 return 前面（或分成新函数）
    names = yolo_model.names  # 类别ID到名称的映射

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # 画框（BGR颜色，线宽2）
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 类别名称
        label = names[class_ids[i]] if class_ids[i] in names else f"id:{class_ids[i]}"
        
        # 加类别文本（左上角）
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # ✅ 打印中心点和类别名称
        print(f"🔹 Center: {centers[i]}, Class: {label}")


    # 保存带框图像
    yolo_output_path = input_path.replace(".png", "_boxed.png")
    cv2.imwrite(yolo_output_path, image_bgr)
    print("✅ YOLO 检测完成，共识别到", len(boxes), "个物体")
    print("centers: ", centers)
    print(f"✅ 检测结果图已保存：{yolo_output_path}")

    return image_rgb, boxes, centers, class_ids, yolo_output_path


# ====== main 函数测试 ======
if __name__ == "__main__":
    input_path = "test.png"
    yolo_path = "yolo_train/train_data_split/runs/detect/train/weights/best.pt"
    detect_fruits(input_path, yolo_path)