import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
# Initial
gs.init(backend=gs.gpu)
# Scene build
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, 3, 1),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)
plane = scene.add_entity(gs.morphs.Plane())
mycobot = scene.add_entity(
    gs.morphs.MJCF(file="/home/ubuntu1/models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper.xml")
)
scene.build()

# Joint set
jnt_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
gripper_names = [
    "gripper_left2_to_gripper_left1",
    "gripper_base_to_gripper_left2",
    "gripper_base_to_gripper_left3",
    "gripper_right2_to_gripper_right1",
    "gripper_base_to_gripper_right2",
    "gripper_base_to_gripper_right3",
]
dofs_idx = [mycobot.get_joint(name).dof_idx_local for name in jnt_names]
gripper_dofs_idx = [mycobot.get_joint(name).dof_idx_local for name in gripper_names]

############ Set control gains ############
# position
mycobot.set_dofs_kp(kp=np.array([4500, 4500, 3500, 3500, 2000, 2000]), dofs_idx_local=dofs_idx)
# velocity
mycobot.set_dofs_kv(kv=np.array([450, 450, 350, 350, 200, 200]), dofs_idx_local=dofs_idx)
# force
mycobot.set_dofs_force_range(
    lower=np.array([-87, -87, -87, -87, -12, -12]),
    upper=np.array([87, 87, 87, 87, 12, 12]),
    dofs_idx_local=dofs_idx,
)

# Hard reset
for i in range(150):
    if i < 50:
        mycobot.set_dofs_position(np.array([0, -0.5, 0.5, -0.5, 0.5, 0]), dofs_idx)
    elif i < 100:
        mycobot.set_dofs_position(np.array([-1, 1, -0.5, 0.5, -1, 1]), dofs_idx)
    else:
        mycobot.set_dofs_position(np.array([0, 0, 0, 0, 0, 0]), dofs_idx)

    scene.step()

############ PD control ############
for i in range(1250):
    if i == 0:
        mycobot.control_dofs_position(np.array([ 0, -0.5, 0.5, -0.5, 0.5, 0]), dofs_idx)
    elif i == 250:
        mycobot.control_dofs_position(np.array([-1, 1, -0.5, 0.5, -1, 1]), dofs_idx)
    elif i == 500:
        mycobot.control_dofs_position(np.array([0, 0, 0, 0, 0, 0]), dofs_idx)
    elif i == 750:
        mycobot.control_dofs_position(np.array([0, 0, 0, 0, 0, 0])[1:], dofs_idx[1:])
        mycobot.control_dofs_velocity(np.array([1, 0, 0, 0, 0, 0])[:1], dofs_idx[:1])
    elif i == 1000:
        mycobot.control_dofs_position(np.array([0, 0, 0, 0, 0, 0]), dofs_idx)
        mycobot.control_dofs_force(np.array([0, 0, 1, 0, 0, 0]), dofs_idx)

    print("Control force:", mycobot.get_dofs_control_force(dofs_idx))
    print("Internal force:", mycobot.get_dofs_force(dofs_idx))

    scene.step()

############ Gripper control############
for i in range(500):
    if i == 0:
        mycobot.control_dofs_position(np.array([0.04, 0.04, 0.04, -0.04, -0.04, -0.04]), gripper_dofs_idx)
    elif i == 250:
        mycobot.control_dofs_position(np.array([-0.02, -0.02, -0.02, 0.02, 0.02, 0.02]), gripper_dofs_idx)

    scene.step()
