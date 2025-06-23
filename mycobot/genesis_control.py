import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np
from scene import create_scene
from constraints_sim import WeldConstraintSim

scene, mycobot, fruits, bins = create_scene()

motors_dof = np.arange(6)

gripper_left_dofs = [
    mycobot.get_joint("gripper_left2_to_gripper_left1").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_left2").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_left3").dof_idx_local,
]
gripper_right_dofs = [
    mycobot.get_joint("gripper_right2_to_gripper_right1").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_right2").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_right3").dof_idx_local,
]

gripper_opening = np.array([0.1] * 3)
gripper_base = mycobot.get_link("gripper_base")
end_effector = mycobot.get_link("tcp")

apple = fruits["apple"]
banana = fruits["banana"]
redBin = bins["redBin"]
yellowBin = bins["yellowBin"]

apple_pos = np.array(apple.get_pos().tolist())
banana_pos = np.array(banana.get_pos().tolist())

def grasp_and_place(obj_pos, drop_pos):
    offset = np.array([0, -0.02, 0])
    quat = np.array([0, 1, 0, 0])

    # 1. 预抓取：末端对准物体上方
    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.15]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=150)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        scene.step()

    # 2. 下探靠近
    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.1]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        scene.step()
    for _ in range(100):
        scene.step()

    # 3. 闭合夹爪夹住
    mycobot.control_dofs_position(-gripper_opening, gripper_left_dofs)
    mycobot.control_dofs_position(gripper_opening, gripper_right_dofs)
    for _ in range(50):
        scene.step()

    # 4. 抬起
    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.1]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        scene.step()

    # 5. 移动到投放点
    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=drop_pos + offset + np.array([0, 0, 0.5]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=120)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        scene.step()

    # 6. 降低一点
    qpos = mycobot.inverse_kinematics(
        link=end_effector,
        pos=drop_pos + offset + np.array([0, 0, 0.42]),
        quat=quat
    )
    path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
    for waypoint in path:
        mycobot.control_dofs_position(waypoint[:-6], motors_dof)
        scene.step()

    # 7. 张开夹爪释放
    mycobot.control_dofs_position(gripper_opening, gripper_left_dofs)
    mycobot.control_dofs_position(-gripper_opening, gripper_right_dofs)
    for _ in range(100):
        scene.step()


red_bin_pos = np.array(redBin.get_pos().tolist())
yellow_bin_pos = np.array(yellowBin.get_pos().tolist())

# 搬运苹果
grasp_and_place(apple_pos, red_bin_pos)

# 搬运香蕉
grasp_and_place(banana_pos, yellow_bin_pos)

# 持续仿真（不结束）
while True:
    scene.step()