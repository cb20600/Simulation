import json
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np
from desk4 import create_scene
from gripper_utils import init_gripper_controller, open_gripper, close_gripper

# 初始化场景
scene, xarm7, fruits, bins, camera = create_scene(enable_gui=True)

# 获取关节索引
arm_dof = np.arange(7)

# # 假设最后6个自由度是夹爪（本体不包含它们）
# gripper_dofs = np.arange(7, 13)
# gripper_opening = np.array([0.06] * 6)

# 获取末端执行器 link
end_effector = xarm7.get_link("xarm_gripper_base_link")

def grasp_and_place(obj_pos, drop_pos):
    offset = np.array([0.0, 0.0, 0.0])
    quat = np.array([0, 0, 1, 0])  # 竖直向下抓取
    # 初始化控制器参数
    init_gripper_controller(xarm7)

    # 1. 预抓取位置
    pos1 = obj_pos + offset + np.array([0.0, 0.0, 0.15])
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos1,
        quat=quat
    )
    print("🎯 目标末端位置:", pos1)
    print("🎯 求解到的关节角度:", qpos)

    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=150)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], arm_dof)
        scene.step()
    print("完成1")

    # 2. 向下靠近土豆
    pos2 = pos1 + np.array([0.0, 0.0, -0.03])  # 贴近土豆
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos2,
        quat=quat
    )
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], arm_dof)
        scene.step()
    print("完成2")

    # 3. 闭合夹爪夹住物体
    print("闭合夹爪")
    # close_gripper(xarm7, scene)  # 确保夹爪完全闭合

    close_gripper(xarm7, scene, contact_entity=fruits["potato"])


    # 检查是否成功夹住物体
    # pos3 = xarm7.get_dofs_position(dofs_idx_local=gripper_dofs)
    # print("夹爪位置:", pos3.detach().cpu().numpy())

    # 确保夹住物体后抬起
    pos4 = pos2 + np.array([0.0, 0.0, 0.3])
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos4,
        quat=quat
    )
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], arm_dof)
        scene.step()
    print("完成4")

    # 4. 移动到放置点上方
    pos5 = drop_pos + np.array([0.0, 0.0, 0.3])
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos5,
        quat=quat
    )
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], arm_dof)
        scene.step()
    print("完成5")

    # 5. 下降放置
    pos6 = pos5 + np.array([0, 0, 0])
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos6,
        quat=quat
    )
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], arm_dof)
        scene.step()
    print("完成6")

    # 6. 张开夹爪释放
    open_gripper(xarm7, scene)  # 确保夹爪打开

    # 7. 抬起回收
    pos8 = pos5
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos8,
        quat=quat
    )
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], arm_dof)
        scene.step()
    print("完成7")



if __name__ == "__main__":
    # ✅ 读取 JSON
    json_path = "imgs/sim_fruit_from_camera.json"
    with open(json_path, "r") as f:
        objects = json.load(f)

    # ✅ 提取香蕉和土豆的位置
    banana_coord = None
    potato_coord = None

    for obj in objects:
        if obj["class"] == "potato":
            potato_coord = np.array(obj["coord_3d"])
        elif obj["class"] == "banana":
            banana_coord = np.array(obj["coord_3d"])
        

    if banana_coord is None or potato_coord is None:
        raise ValueError("未找到 banana 或 potato 的位置信息")

    # ✅ 计算土豆左边的位置（假设 y 轴是左右方向）
    offset = np.array([0.0, -0.2, 0.0])  # 向左偏移
    drop_coord = potato_coord + offset
    print("香蕉位置：", banana_coord, "放置位置：", drop_coord)
        # 3. 定义初始和目标关节角
    pos_init   = np.array([0, 0, 0, 1, 0, 0.5, 0])
    for _ in range(100):
        scene.step()

    # 6. 重置到初始姿态
    for i in range(100):
        xarm7.set_dofs_position(pos_init, arm_dof)
        scene.step()
    # ✅ 执行抓取与放置
    grasp_and_place(potato_coord, drop_coord)
