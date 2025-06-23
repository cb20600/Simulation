import json
import os

import torch
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np
from desk4 import create_scene 
from gripper_utils import init_gripper_controller, close_gripper, open_gripper,move_gripper_to


if __name__ == "__main__":
    
    enable_gui=True
    scene, xarm7, fruits, bins, camera = create_scene(enable_gui)
    print("等待物理系统稳定...")

    for _ in range(100):
        scene.step()
    all_dof = np.arange(13)
    motors_dof = np.arange(7)
    gripper_dof = np.arange(7, 13)
    fingers_dof = np.arange(11,13)

    # lime = fruits["lime"]
    # print("lime:", lime.get_pos())
    # jnt_names = [
    #     "joint1", "joint2", "joint3",
    #     "joint4", "joint5", "joint6", "joint7"
    # ]
    # dofs_idx = [xarm7.get_joint(name).dof_idx_local for name in jnt_names]

    # 2. PID & 力矩限制设置
    xarm7.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        dofs_idx_local=motors_dof#dofs_idx
    )
    xarm7.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200]),
        dofs_idx_local=motors_dof#dofs_idx
    )
    xarm7.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
        upper=np.array([ 87,  87,  87,  87,  12,  12,  12]),
        dofs_idx_local=motors_dof#dofs_idx
    )

    # 3. 定义初始和目标关节角
    pos_init   = np.array([0, 0, 0, 1, 0, 0.5, 0])


    print("等待物理系统稳定...")
    
    for _ in range(100):
        scene.step()

    # 6. 重置到初始姿态
    for i in range(100):
        xarm7.set_dofs_position(pos_init, motors_dof)#dofs_idx)
        scene.step()

    # if enable_gui==True:
    #     camera.render(rgb=True)

    # 设置末端执行器为夹爪基座
    end_effector = xarm7.get_link("xarm_gripper_base_link")
    # 设置显示该 link 的坐标轴
    end_pos = end_effector.get_pos().cpu().numpy()
    print("末端执行器位置：",end_pos)

    obj_pos = np.array([0.14, 0.1, 0.38])
    offset = np.array([0.053, -0.000, 0.15])
    quat = np.array([0, 0.5, 1, 0])

    pos1=obj_pos + offset + np.array([0.0, 0.0, 0.15])

    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        # pos=obj_pos + offset + np.array([0.0, 0.0, 0.15]),
        pos = pos1,
        quat = quat
    )

    qpos[7:] = 0.02
    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)

    for waypoint in path:
        xarm7.control_dofs_position(waypoint)
        scene.step()

    for i in range(100):
        scene.step()
    print("预抓取位置")

    # reach
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=obj_pos + offset + np.array([0.0, 0.0, 0.03]),
        # pos = obj_pos + offset,
        quat=quat
    )

    path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
    for waypoint in path:
        xarm7.control_dofs_position(waypoint[:7], motors_dof)
        scene.step()
    for i in range(100):
        scene.step()
    print("reach")

    # grasp
    qpos[7:] = 0.4
    xarm7.control_dofs_position(qpos)
    xarm7.control_dofs_force(np.array([-1, -1]), fingers_dof)
    scene.step()
    for i in range(100):
        scene.step()
    print("grasp")

    # lift
    qpos = xarm7.inverse_kinematics(
        link=end_effector,
        pos=pos1,
        quat=quat
    )
    xarm7.control_dofs_position(qpos[:7], motors_dof)
    for i in range(100):
        scene.step()
    print("lift")



    '''
    # base_pos = xarm7.get_link("xarm_gripper_base_link").get_pos()
    # left_finger_pos = xarm7.get_link("left_finger").get_pos()
    # right_finger_pos = xarm7.get_link("right_finger").get_pos()

    # print("gripper_base_pos     = [{:.3f}, {:.3f}, {:.3f}]".format(*base_pos))
    # print("left_finger_pos      = [{:.3f}, {:.3f}, {:.3f}]".format(*left_finger_pos))
    # print("right_finger_pos     = [{:.3f}, {:.3f}, {:.3f}]".format(*right_finger_pos))

    # grasp_point = (left_finger_pos + right_finger_pos) / 2
    # gripper_offset = grasp_point - base_pos

    # print("⚙️ gripper_offset     = [{:.3f}, {:.3f}, {:.3f}]".format(*gripper_offset))
    '''
    # # 设置目标位置和方向（竖直向下）
    # target_pos = np.array([0.20, 0.0, 1])
    # target_quat = np.array([0, 0, 1, 0])  # Z轴朝下，姿态可微调

    # # 使用逆运动学求解目标姿态的关节位置
    # qpos = xarm7.inverse_kinematics(
    #     link=end_effector,
    #     pos=target_pos,
    #     quat=target_quat,
    # )

    # # 设置夹爪张开（假设末6个自由度是夹爪）
    # qpos[7:] = 0.04

    # # 生成运动路径
    # path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=200)

    # # 执行路径
    # if enable_gui:
    #     for waypoint in path:
    #         xarm7.control_dofs_position(waypoint)
    #         scene.step()
    #         camera.render(rgb=True)
    # else:
    #     for waypoint in path:
    #         xarm7.control_dofs_position(waypoint)
    #         scene.step()

    # # control_xarm7(scene, xarm7, cam_attachment, camera)
    # # print(camera.transform)
    # for i in range(5000):
    #     scene.step()
    # print("✅ 控制结束")
    # 
