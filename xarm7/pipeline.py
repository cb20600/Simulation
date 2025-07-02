# from utils.pose_capture import simulate_and_capture_scene
from utils.yolo_utils import detect_fruits
from utils.sam2_utils import segment_with_sam2
from utils.coordinate import annotate_and_get_3d_coords
from utils.save_json import save_json_from_detection
from utils.rotate import extract_grasp_infos, grasp_angle_to_quaternion

if __name__ == "__main__": 
    '''
    1. 调用LLM读取用户指令(需要模糊功能识别拼写错误等): LLM输出子任务序列
    2. 创建场景,执行场景全览运动,调用摄像头拍摄桌面图像: 输出为RGB-D图像
    3. 使用yolo_utils.py进行识别: image_rgb, boxes, yolo_centers, class_ids
    4. SAM2处理(sam2_utils.py): out_masks, centers, save_path. 计算物体主轴副轴方向, 给出对应的quat值(rotate.py)
    5. 调用corrdinate.py反投影点云计算3D位置, 并使用save_json.py保存为json文件
    6. LLM读取json文件内容, 并根据子任务序列，生成详细的轨迹点和任务
    7. 调用仿真控制器执行轨迹点序列, 根据quat值和物体宽度调整夹爪角度和开合大小
    '''
    img_path = "imgs/sim_fruit_from_camera.png"
    yolo_path = "checkpoints/best.pt"
    sam_model_path = "checkpoints/sam2_b.pt"
    npz_path = "imgs/sim_fruit_from_camera.npz"

    # ======================= Step 2: Capture Pose =======================
    simulate_and_capture_scene()

    # ======================= Step 3: YOLO Detection =======================
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(
        input_path=img_path,
        yolo_path=yolo_path
    )

    if len(boxes) == 0:
        print("❌ No objects detected by YOLO. Exit!")
        exit()

    # ======================= Step 4: SAM2 Segmentation ====================
    masks, sam_centers = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=boxes,
        sam_model_path=sam_model_path,
        input_image_path=img_path
    )

    # Optional: extract grasp angles and quaternions
    grasp_infos, grasp_img_path = extract_grasp_infos(
        image_rgb=image_rgb,
        masks=masks,
        pixel_to_meter=0.0025,
        output_path=img_path.replace(".png", "_grasp_infos.png")
    )

    quaternions = []
    for info in grasp_infos:
        angle_deg = info["angle_deg"]
        quat = grasp_angle_to_quaternion(angle_deg)
        quaternions.append(quat)
        print(f"🔁 Object #{info['index']} grasp angle = {angle_deg:.2f}°, quaternion = {quat}")

    # ======================= Step 5: 3D Coordinate Projection =============
    coords_3d = annotate_and_get_3d_coords(
        image_path=img_path,
        npz_path=npz_path,
        pixel_points=sam_centers  # Prefer sam_centers for precision
    )

    # Print coordinates
    for i, coord in enumerate(coords_3d):
        if coord is not None:
            print(f"🧭 Object #{i} 3D position: [{coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}]")
        else:
            print(f"⚠️ Object #{i} has no valid 3D coordinate")

    from utils.save_json import save_json_from_detection

    save_json_from_detection(
        yolo_path=yolo_path,
        class_ids=class_ids,
        centers=sam_centers,
        coords_3d=coords_3d,
        widths=[info["width_m"] for info in grasp_infos],
        quaternions=quaternions,
        # gripper_opens=[compute_gripper_open_close(info["width_m"])[0] for info in grasp_infos],
        # gripper_closes=[compute_gripper_open_close(info["width_m"])[1] for info in grasp_infos],
        # output_path=json_path
    )
