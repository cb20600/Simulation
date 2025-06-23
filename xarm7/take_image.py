import json
import os

import torch
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np
from desk4 import create_scene 
from gripper_utils import init_gripper_controller, close_gripper, open_gripper,move_gripper_to

# 初始化场景
scene, xarm7, fruits, bins, camera = create_scene(enable_gui=True)

all_dof = np.arange(13)
arm_dof = np.arange(7)

# gripper_dofs = np.arange(7, 13)
# gripper_open = np.array([0.06] * 6)

end_effector = xarm7.get_link("xarm_gripper_base_link")    

pos_init = np.array([0, 0, 0, 1, 0, 0.5, 0])

# for i in range(100):
#     xarm7.set_dofs_position(pos_init, arm_dof)
#     scene.step()   
# print("初始姿态完成")

# def control_gripper(gripper_dofs, position, steps=200):
#     """控制夹爪张开或闭合"""
#     # 这里position是6维，例如：np.array([0.06]*6) 表示张开
#     for _ in range(steps):
#         xarm7.set_dofs_position(position, gripper_dofs)
#         scene.step()


offset = np.array([0.0, 0.0, 0.15])
quat = np.array([0, 0, 1, 0])  # 竖直向下抓取

# 1. 预抓取位置
obj_pos = (0.2, 0.05, 0.42)

pos1=obj_pos + offset + np.array([0.0, 0.0, 0.15])
qpos = xarm7.inverse_kinematics(
    link=end_effector,
    # pos=obj_pos + offset + np.array([0.0, 0.0, 0.15]),
    pos = pos1,
    quat=quat
)

qpos[7:] = 0.25
print(qpos)
qpos[7:] = torch.tensor([0.45] * 6, device=qpos.device, dtype=qpos.dtype)
print(qpos)
path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=150)
for waypoint in path:
    xarm7.control_dofs_position(waypoint, all_dof)
    scene.step()

for i in range(100):
    scene.step()

print("完成1",)
init_gripper_controller(xarm7)
# 在抓取前打开夹爪
# open_gripper(xarm7, scene, 0.4,0.0)

# 2. 向下靠近
pos2 = pos1 + np.array([0.0, 0.0, -0.15])
qpos = xarm7.inverse_kinematics(
    link=end_effector,
    # pos=obj_pos + offset + np.array([0.0, 0.0, 0.03]),
    pos = obj_pos + offset,
    quat=quat
)

path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
for waypoint in path:
    xarm7.control_dofs_position(waypoint[:7], arm_dof)
    scene.step()
for i in range(100):
    scene.step()
print("完成2")

# qpos[7:] = 0.35
# path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
# for waypoint in path:
#     xarm7.control_dofs_position(waypoint, all_dof)
#     scene.step()

# gripper_open = np.array([0.35] * 6)
# control_gripper(gripper_dofs, gripper_open)
# final = xarm7.get_dofs_position(dofs_idx_local=gripper_dofs)
# print(f"✅ Gripper complete. Final position:", final.cpu().numpy())

# gripper_close = np.array([-0.35] * 6)
# control_gripper(gripper_dofs, gripper_close)

# # 3. 抬起
# qpos = xarm7.inverse_kinematics(
#     link=end_effector,
#     # pos=obj_pos + offset + np.array([0.0, 0.0, 0.03]),
#     pos = pos1,
#     quat=quat
# )
# path = xarm7.plan_path(qpos_goal=qpos, num_waypoints=100)
# for waypoint in path:
#     xarm7.control_dofs_position(waypoint[:7], arm_dof)
#     scene.step()
# for _ in range(200):
#     scene.step()
# print("完成3")

# allow robot to reach the last waypoint
# for i in range(100):
#     scene.step()
# print("完成4")

# import os
# os.environ['PYOPENGL_PLATFORM'] = 'glx'

# import numpy as np
# import genesis as gs
# from desk4 import create_scene
# from gripper_utils import init_gripper_controller, open_gripper, close_gripper


# # 初始化控制器参数
# init_gripper_controller(xarm7)

# # 打开夹爪
# open_gripper(xarm7, scene)

# # 闭合夹爪
# close_gripper(xarm7, scene)

# print("✅ Gripper test done.")

# 接近物体后慢慢闭合
# close_gripper(xarm7, scene, 0, 0.4)

move_gripper_to(xarm7, scene, 0.5)
move_gripper_to(xarm7, scene, 0)