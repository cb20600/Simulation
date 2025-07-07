# from utils.pose_capture import simulate_and_capture_scene
from utils.yolo_utils import detect_fruits
from utils.sam2_utils import segment_with_sam2
from utils.coordinate import annotate_and_get_3d_coords
from utils.save_json import save_json_from_detection
from utils.rotate import extract_grasp_infos, simple_quaternion_from_angle
from utils.desk4 import create_scene
from utils.gripper_utils import GripperController

from GPT import parser
from pose_capture import capture
from execute_trajectory import execute_trajectory

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

if __name__ == "__main__": 
    '''
    1. 创建场景,执行场景全览运动,调用摄像头拍摄桌面图像: 输出为RGB-D图像
    2. 使用yolo_utils.py进行识别: image_rgb, boxes, yolo_centers, class_ids
    3. SAM2处理(sam2_utils.py): out_masks, centers, save_path. 计算物体主轴副轴方向, 给出对应的quat值(rotate.py)
    4. 调用LLM读取用户指令(需要模糊功能识别拼写错误等): LLM输出子任务序列
    5. 调用corrdinate.py反投影点云计算3D位置, 并使用save_json.py保存为json文件
    6. LLM读取json文件内容, 并根据子任务序列，生成详细的轨迹点和任务
    7. 调用仿真控制器执行轨迹点序列, 根据quat值和物体宽度调整夹爪角度和开合大小
    '''
    enable_gui = True
    yolo_model_path = "checkpoints/best_yolo.pt"
    sam2_model_path = "checkpoints/sam2_b.pt"
    path_base = "imgs/sim_fruit_from_camera"
    img_path = f"{path_base}.png"
    npz_path = f"{path_base}.npz"
    json_path = f"{path_base}.json"


    # ======================= Step 1: Capture Pose =======================
    import numpy as np
    import cv2
    import os
    from utils.desk4 import create_scene


    """
    控制 xArm7 到达指定姿态，并保存 RGB 图像和点云信息

    参数:
        scene: Genesis 场景对象
        xarm7: xArm7 机械臂对象
        camera: 相机对象
        enable_gui (bool): 是否开启渲染
        img_path (str): 保存 RGB 图像的路径
        npz_path (str): 保存点云的路径

    返回:
        dict: 包含 'img_path' 和 'npz_path'
    """
    enable_gui=True
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui)
    print("等待物理系统稳定...")
    for _ in range(100):
        scene.step()

    jnt_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dofs_idx = [xarm7.get_joint(name).dof_idx_local for name in jnt_names]

    # PID 和力矩限制
    xarm7.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]), dofs_idx_local=dofs_idx)
    xarm7.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]), dofs_idx_local=dofs_idx)
    xarm7.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12]),
        dofs_idx_local=dofs_idx
    )

    # 初始姿态
    pos_init = np.array([0, 0, 0, 1, 0, 0.5, 0])
    for _ in range(100):
        scene.step()
    for _ in range(100):
        xarm7.set_dofs_position(pos_init, dofs_idx)
        scene.step()

    # 获取末端执行器和夹爪位置
    end_effector = xarm7.get_link("xarm_gripper_base_link")
    base_pos = end_effector.get_pos()
    left_finger_pos = xarm7.get_link("left_finger").get_pos()
    right_finger_pos = xarm7.get_link("right_finger").get_pos()
    gripper_offset = (left_finger_pos + right_finger_pos) / 2 - base_pos

    # 设置目标位置和姿态（Z轴向下）
    target_pos = np.array([0.20, 0.0, 0.85])
    target_quat = np.array([0, 0, 1, 0])

    # 求解逆解 + 设置夹爪张开
    qpos = xarm7.inverse_kinematics(link=end_effector, pos=target_pos, quat=target_quat)
    qpos[7:] = 0.04

    # 路径规划 + 执行
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=200)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint)
        scene.step()
        if enable_gui:
            camera.render(rgb=True)

    for _ in range(50):
        scene.step()
    print("✅ 控制结束")

    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    if enable_gui:
        # 保存 RGB 图像
        rgb_img, _, _, _ = camera.render(rgb=True)
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, rgb_bgr)
        print(f"✅ 图像已保存到: {img_path}")

        # 保存点云
        pointcloud, mask_idx = camera.render_pointcloud(world_frame=True)[:2]
        mask = np.zeros(pointcloud.shape[:2], dtype=bool)
        mask[mask_idx] = True
        np.savez(npz_path, pointcloud=pointcloud, mask=mask)
        print(f"✅ 点云和掩码已保存到: {npz_path}")

    # return {"img_path": img_path, "npz_path": npz_path}


    # ======================= Step 2: YOLO Detection =======================
    image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)
    if len(boxes) == 0:
        print("❌ No objects detected by YOLO. Exit!")
        exit()

    # ======================= Step 3: SAM2 Segmentation & Grasp Info Extraction ====================
    masks, sam_centers, result_path = segment_with_sam2(
        image_rgb=image_rgb,
        boxes=boxes,
        sam_model_path=sam2_model_path,
        input_image_path=img_path
    )
    grasp_infos, grasp_img_path = extract_grasp_infos(
        image_rgb=image_rgb,
        masks=masks,
        pixel_to_meter=None,
        output_path=f"{path_base}_grasp_visual.png"
    )

    quat = [simple_quaternion_from_angle(info["angle_deg"]) for info in grasp_infos]
    widths = [info["width"] for info in grasp_infos]

    # print("✅ 抓取信息：")
    # for info, quat in zip(grasp_infos, quat):
    #     print(f"# {info['index']}: angle={info['angle_deg']:.2f}°, width={info['width']:.4f}m, quat={quat}")

    # ======================= Step 4: 3D Coordinate Projection =============
    coords_3d = annotate_and_get_3d_coords(
        image_path=img_path,
        npz_path=npz_path,
        pixel_points=sam_centers
    )

    for i, coord in enumerate(coords_3d):
        if coord is not None:
            print(f"🧭 Object #{i} 3D position: [{coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}]")
        else:
            print(f"⚠️ Object #{i} has no valid 3D coordinate")

    # ======================= Step 5: Save JSON ============================
    # 判断是否使用 YOLO 类别映射
    if class_ids is not None and yolo_model_path:
        save_json_from_detection(
            class_ids=class_ids,
            centers=sam_centers,
            coords_3d=coords_3d,
            widths=widths,
            quaternions=quat,
            yolo_path=yolo_model_path,
            output_path=json_path
        )
    else:
        # 自定义类别映射：object_0, object_1 ...
        custom_names = {i: f"object_{i}" for i in range(len(sam_centers))}
        save_json_from_detection(
            class_ids=[i for i in range(len(sam_centers))],
            centers=sam_centers,
            coords_3d=coords_3d,
            widths=widths,
            quaternions=quat,
            class_names=custom_names,
            output_path=json_path
        )

    # ======================= Step 6: LLM Command Parse =======================
    # 初始化 API
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 接收用户指令
    user_input = input("Enter your robot instruction: ")
    task_description = parser(user_input)

    # ======================= Step 6.5: 保存子任务文本 =======================
    with open("llm_subtasks.txt", "w") as f:
        f.write(task_description)

    # ======================= Step 7: LLM Trajectory Planning =======================
    from GPT import trajectory_plan  # 如果 trajectory_plan 定义在 GPT.py
    trajectory_result = trajectory_plan(
        subtask_txt_path="llm_subtasks.txt",
        grasp_json_path=json_path
    )
    # 初始化控制器
    gripper = GripperController(xarm7, scene)
    # 执行轨迹控制（前提是你已生成 trajectory_plan.json）
    execute_trajectory(
        xarm7=xarm7,
        gripper=gripper,
        trajectory_json_path="trajectory_plan.json",
        scene=scene,  # 如果你有 Genesis 场景
        wait_steps=40
    )

'''
    # # ======================= Step 3: YOLO Detection =======================

    # # 使用你原来的 YOLO 检测函数
    # image_rgb, boxes, yolo_centers, class_ids, yolo_output_path = detect_fruits(img_path, yolo_model_path)
    # if len(boxes) == 0:
    #     print("❌ No objects detected by YOLO. Exit!")
    #     exit()

    # print("yolo中心点：", yolo_centers)

    # # ======================= Step 4: SAM2 Segmentation ====================
    # masks, sam_centers, result_path = segment_with_sam2(image_rgb, boxes, sam2_model_path, img_path)
    # print("SAM2中心点：", sam_centers)

    # grasp_infos, yolo_img_path = extract_grasp_infos(
    # image_rgb=image_rgb,
    # masks=masks,
    # pixel_to_meter=0.0025,  # 可选：像素转换为米的比例
    # output_path="grasp_infos_visual.png"
    # )

    # # 输出结果
    # print("✅ 抓取信息如下：")
    # for info in grasp_infos:
    #     print(info)
    # for info in grasp_infos:
    #     angle = info["angle_deg"]
    #     quat = grasp_angle_to_quaternion(angle)
    #     print(f"物体 #{info['index']} 四元数姿态: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]")


    # print("✅ SAM2 可视化图已保存至：", output_img_path)

    # masks, sam_centers, sam2_result = segment_with_sam2(
    #     image_rgb=image_rgb,
    #     boxes=boxes,
    #     sam_model_path=sam2_model_path,
    #     input_image_path=img_path
    # )

    # # Optional: extract grasp angles and quaternions
    # grasp_infos, grasp_img_path = extract_grasp_infos(
    #     image_rgb=image_rgb,
    #     masks=masks,
    #     pixel_to_meter=0.0025,
    #     output_path=img_path.replace(".png", "_grasp_infos.png")
    # )

    # quaternions = []
    # for info in grasp_infos:
    #     angle_deg = info["angle_deg"]
    #     quat = grasp_angle_to_quaternion(angle_deg)
    #     quaternions.append(quat)
    #     print(f"🔁 Object #{info['index']} grasp angle = {angle_deg:.2f}°, quaternion = {quat}")

    # # ======================= Step 5: 3D Coordinate Projection =============
    # coords_3d = annotate_and_get_3d_coords(
    #     image_path=img_path,
    #     npz_path=npz_path,
    #     pixel_points=sam_centers  # Prefer sam_centers for precision
    # )

    # # Print coordinates
    # for i, coord in enumerate(coords_3d):
    #     if coord is not None:
    #         print(f"🧭 Object #{i} 3D position: [{coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}]")
    #     else:
    #         print(f"⚠️ Object #{i} has no valid 3D coordinate")

    # from utils.save_json import save_json_from_detection

    # save_json_from_detection(
    #     yolo_path=yolo_model_path,
    #     class_ids=class_ids,
    #     centers=sam_centers,
    #     coords_3d=coords_3d,
    #     widths=[info["width_m"] for info in grasp_infos],
    #     quaternions=quaternions,
    #     # gripper_opens=[compute_gripper_open_close(info["width_m"])[0] for info in grasp_infos],
    #     # gripper_closes=[compute_gripper_open_close(info["width_m"])[1] for info in grasp_infos],
    #     # output_path=json_path
    # )
'''