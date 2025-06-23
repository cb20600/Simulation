import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
import numpy as np

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, 3, 1),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.03, 0.03, 0.03),
        pos=(0.4, 0.0, 0.02),
    )
)
mycobot = scene.add_entity(
    gs.morphs.MJCF(file="/home/ubuntu1/models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper.xml"),
)

########################## build ##########################
scene.build()

motors_dof = np.arange(6)
gripper_left_dofs = [mycobot.get_joint("gripper_left2_to_gripper_left1").dof_idx_local, mycobot.get_joint("gripper_base_to_gripper_left2").dof_idx_local, mycobot.get_joint("gripper_base_to_gripper_left3").dof_idx_local]
gripper_right_dofs = [mycobot.get_joint("gripper_right2_to_gripper_right1").dof_idx_local, mycobot.get_joint("gripper_base_to_gripper_right2").dof_idx_local, mycobot.get_joint("gripper_base_to_gripper_right3").dof_idx_local]
end_effector = mycobot.get_link("gripper_base")
gripper_opening = np.array([0.04] * 3)

########################## move to pre-grasp pose ##########################
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.3, 0.00, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
# gripper open pos
path = mycobot.plan_path(
    qpos_goal=qpos,
    num_waypoints=200, # 2s duration
)
# execute the planned path
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()
# allow robot to reach the last waypoint
for i in range(100):
    scene.step()

# # 控制夹爪
mycobot.control_dofs_position(gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(-gripper_opening, gripper_right_dofs)
for i in range(50):
    scene.step()

# reach
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.3, 0.0, 0.05]),
    quat=np.array([0, 1, 0, 0]),
)
path = mycobot.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 2s duration
)
# execute the planned path
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()
for i in range(150):
    scene.step()

mycobot.control_dofs_position(-gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(gripper_opening, gripper_right_dofs)
for i in range(100):
    scene.step()

# lift
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.45, 0.0, 0.45]),
    quat=np.array([0, 1, 0, 0]),
)
path = mycobot.plan_path(
    qpos_goal=qpos,
    num_waypoints=150,  # 2s duration
)
# execute the planned path
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()
for i in range(200):
    scene.step()

mycobot.control_dofs_position(qpos[:-6], motors_dof)
for i in range(200):
    scene.step()

# joint_limits_lower, joint_limits_upper = mycobot.get_dofs_limit()
# print("Low limit of joint:", joint_limits_lower)
# print("Hight limit of joint:", joint_limits_upper)
# print("qpos_start:", qpos)