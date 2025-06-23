import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np
from constraints_sim import WeldConstraintSim
# Initialize Genesis engine
gs.init(backend=gs.gpu)

# Create the simulation scene
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
       #  camera_pos=(1.2, 1.2, 0.8),
        camera_pos=(0, 1.2, 0.6),
        camera_lookat=(0.0, 0.0, 0.15),
        camera_fov=45,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

# Add the ground plane
scene.add_entity(gs.morphs.Plane())

# Add table top
scene.add_entity(gs.morphs.Box(
    size=(0.6, 0.4, 0.02),
    pos=(0.0, 0.0, 0.15),
    fixed=True
))

# Add table legs
for dx in [-0.275, 0.275]:
    for dy in [-0.175, 0.175]:
        scene.add_entity(gs.morphs.Box(
            size=(0.05, 0.05, 0.15),
            pos=(dx, dy, 0.075),
            fixed=True
        ))

# Add the medicine bottle (initially fixed)
cube = scene.add_entity(
    gs.morphs.Cylinder(
    height = 0.05,
    radius = 0.015,
    pos=(0.19, 0.0, 0.185),
    # size=(0.03, 0.03, 0.05),
    collision=False,
    visualization=True,
    fixed=True,
    )
)
# Add the MyCobot
mycobot = scene.add_entity(gs.morphs.MJCF(
    file="/home/skyler/code/Genesis-main/models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper.xml",
    pos=(-0.10, 0.0, 0.16),
))

# Build the scene
scene.build()

# Get joint info for controlling the arm and gripper
motors_dof = np.arange(6)

gripper_left_dofs = [
    mycobot.get_joint("gripper_left2_to_gripper_left1").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_left2").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_left3").dof_idx_local
]
gripper_right_dofs = [
    mycobot.get_joint("gripper_right2_to_gripper_right1").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_right2").dof_idx_local,
    mycobot.get_joint("gripper_base_to_gripper_right3").dof_idx_local
]

# End-effector link
end_effector = mycobot.get_link("gripper_base")
# Gripper open configuration
gripper_opening = np.array([0.05] * 3)

# Move to grasp position
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.2, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=300)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    scene.step()
for _ in range(50):
    scene.step()

# Open the gripper
mycobot.control_dofs_position(gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(-gripper_opening, gripper_right_dofs)
for _ in range(50):
    scene.step()

# Close gripper to grasp the bottle
mycobot.control_dofs_position(-gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(gripper_opening, gripper_right_dofs)
for _ in range(100):
    scene.step()

# Create weld constraint after grasping (offset to align to gripper center)
weld = WeldConstraintSim(
    robot_entity=mycobot,
    left_link_name="gripper_left3",
    right_link_name="gripper_right3",
    cube_entity=cube,
    offset=np.array([0.0, 0.0, -0.07])
)

# Unfix the cube so it can move
cube.fixed = False
# Initial attach
weld.step()
for _ in range(10):
    scene.step()

# Lift the bottle
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.2, 0.0, 0.4]),
    quat=np.array([0, 1, 0, 0]),
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=50)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    weld.step()
    scene.step()
for _ in range(100):
    weld.step()
    scene.step()

# Move to drop position
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=np.array([-0.4, 0.0, 0.3]),
    quat=np.array([0, 1, 0, 0]),
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=200)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    weld.step()
    scene.step()
for _ in range(50):
    weld.step()
    scene.step()

# Lower bottle
qpos = mycobot.inverse_kinematics(
    link=end_effector,
    pos=np.array([-0.4, 0.0, 0.1]),
    quat=np.array([0, 1, 0, 0]),
)
path = mycobot.plan_path(qpos_goal=qpos, num_waypoints=100)
for waypoint in path:
    mycobot.control_dofs_position(waypoint[:-6], motors_dof)
    weld.step()
    scene.step()
for _ in range(50):
    weld.step()
    scene.step()

# Open gripper to release bottle
mycobot.control_dofs_position(gripper_opening, gripper_left_dofs)
mycobot.control_dofs_position(-gripper_opening, gripper_right_dofs)
for _ in range(200):
    scene.step()

# Optionally re-fix the cube after placing
cube.fixed = True
