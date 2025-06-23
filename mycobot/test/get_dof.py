import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene()

plane = scene.add_entity(
    gs.morphs.Plane(),
)

xarm7 = scene.add_entity(
    # gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    # gs.morphs.URDF(file="/home/ubuntu1/models/mycobot_320_m5_2022/new_mycobot_pro_320_m5_2022_gripper.urdf"),
    # gs.morphs.MJCF(file="/home/skyler/code/Individual_Project/Genesis-main/models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper1.xml"),
    gs.morphs.MJCF(file="models/ufactory_xarm7/xarm7.xml"),
)
# print(dir(franka))
print(f"Total DOF: {xarm7.n_dofs}")
# print("Joint Names:", mycobot.joints)
print("Link Names:", xarm7.links)

# end_effector = mycobot.get_link("gripper_base")
# print(dir(mycobot))

# from inspect import signature
# print(signature(gs.morphs.Box))


for i, link in enumerate(xarm7.links):
    print(f"Link {i}: name = '{link.name}', index = {link.idx}")
for j in xarm7.joints:
    print(f"Joint: {j.name}, DOF Index: {j.dof_idx_local}")



