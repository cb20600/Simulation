import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
gs.init(backend=gs.gpu)
import math
import numpy as np
from scene import create_scene

# 初始化场景和实体
scene, mycobot, fruits, markers = create_scene()
apple = fruits["apple"]
banana = fruits["banana"]
strawberry = fruits["strawberry"]
tcp_marker = markers["tcp_marker"]
base_marker = markers["base_marker"]

# 获取运动关节
motors_dof = np.arange(6)
# 夹爪自由度（可根据你的模型改名）
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

# 获取目标水果位置
apple_pos = np.array(apple.get_pos().tolist())
print(apple_pos)
banana_pos = np.array(banana.get_pos().tolist())
offset = np.array([0, -0.02, 0])

# 1. pregraspe
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=apple_pos + offset + np.array([0.0, 0.0, 0.15]),
    quat=np.array([0, 1, 0, 0])
)
print(11111111111111)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=150)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)

    scene.step()

# 2. 下探靠近
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=apple_pos + offset + np.array([0.0, 0.0, 0.1]),
    quat=np.array([0, 1, 0, 0])
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=100)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()
for _ in range(100):
    scene.step()


# 3. 闭合夹爪夹住苹果
print(33333333333)
mycobot.control_dofs_position(-gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(gripper_opening, gripper_right_dofs)
for _ in range(100):
    scene.step()

# 4. 抬起
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=apple_pos + offset + np.array([0.0, 0.0, 0.15]),
    quat=np.array([0, 1, 0, 0])
)
print(4444444444444)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()

# 5. 移动到投放点（靠近垃圾桶）
drop_pos = np.array([0.6, -0.15, 0.55])
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=drop_pos + offset,
    quat=np.array([0, 1, 0, 0])
)
print(55555555555)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=120)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()

# 6. 降低一点
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=drop_pos + offset + np.array([0, 0, -0.06]),
    quat=np.array([0, 1, 0, 0])
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()

# 7. 张开夹爪释放
mycobot.control_dofs_position(gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(-gripper_opening, gripper_right_dofs)
scene.step()
for _ in range(100):
    scene.step()


# 2. Banana Grasp
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=banana_pos + offset + np.array([0.0, 0.0, 0.15]),
    quat=np.array([0, 1, 0, 0])
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=150)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()

# 2. 下探靠近
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=banana_pos + offset + np.array([0.0, 0.0, 0.1]),
    quat=np.array([0, 1, 0, 0])
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=100)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()
for _ in range(100):
    scene.step()

# 3. 闭合夹爪夹住苹果
mycobot.control_dofs_position(-gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(gripper_opening, gripper_right_dofs)
for _ in range(100):
    scene.step()

# 4. 抬起
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=banana_pos + offset + np.array([0.0, 0.0, 0.15]),
    quat=np.array([0, 1, 0, 0])
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()

# 5. 移动到投放点（靠近垃圾桶）
drop_pos = np.array([0.6, 0.15, 0.55])
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=drop_pos + offset,
    quat=np.array([0, 1, 0, 0])
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=120)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()

# 6. 降低一点
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=drop_pos + offset + np.array([0, 0, -0.06]),
    quat=np.array([0, 1, 0, 0])
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=60)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()

# 7. 张开夹爪释放
mycobot.control_dofs_position(gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(-gripper_opening, gripper_right_dofs)
scene.step()
for _ in range(100):
    scene.step()


# 持续运行仿真
while True:
    scene.step()


# qpos = mycobot.inverse_kinematics(
#     link=end_effector,
#     pos=np.array([0.45, 0.1, 0.5]),
#     quat=np.array([0, 1, 0, 0])
# )
# path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=150)
# for waypoint in path:
#     mycobot.control_dofs_position(waypoint[:-6], motors_dof)
#     scene.step()