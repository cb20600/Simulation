import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
import numpy as np

# Initialize Genesis engine
gs.init(backend=gs.gpu)

# Create the simulation scene
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2, 2, 1),
        # camera_pos=(1.2, 1.2, 0.8),
        # camera_pos=(0, 1.2, 0.6),
        camera_lookat=(0.0, 0.0, 0.15),
        camera_fov=45,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

# Add the ground plane
scene.add_entity(gs.morphs.Plane())

scene.add_entity(gs.morphs.Mesh(
    file="models/components/dining_table_chair.glb",
    pos=(0, 0, 0),
    euler=(90, 0, 0),
    scale=0.4,
    fixed=True,
    convexify=False,
    collision=True,
))

# Add the MyCobot
mycobot = scene.add_entity(gs.morphs.MJCF(
    file="models/mujoco_mycobot-main/mujoco_mycobot-main/mycobot_with_gripper.xml",
    pos=(0.20, 0.0, 0.3)
))


# Add fruits
# Apple
apple = scene.add_entity(gs.morphs.Mesh(
    file="models/components/fruits/apple/10162_Apple_v01_l3.obj",
    pos=(0.35, -0.1, 0.30),
    scale=0.0002,
    fixed=False,     
    # fixed=True, 
    visualization=True,
    collision=True, 
))
# Banana
scene.add_entity(gs.morphs.Mesh(
    file="models/components/fruits/banana.glb",
    pos=(0.35, 0, 0.30),
    euler=(90, 0, 60),
    scale=0.1,
    fixed=False,
    # fixed=True, 
    visualization=True,
    collision=True,
))

orange = scene.add_entity(gs.morphs.Mesh(
    file="models/components/fruits/12204_Fruit_v1_L3.obj",
    pos=(0.45, 0.1, 0.4),
    scale=0.001,
    fixed=True,
    visualization=True,
    collision=True,
))

# Strawberry
scene.add_entity(gs.morphs.Mesh(
    file="models/components/fruits/strawberry.glb",
    pos=(0.35, 0.1, 0.30),
    scale=0.005,
    fixed=False,
    # fixed=True, 
    visualization=True,
    collision=True,
))


# Import Trash Bin
# scene.add_entity(gs.morphs.Mesh(
#     file="models/components/trashbin/Trashbin.glb",
#     pos=(0.55, 0.15, 0.05),
#     euler=(90, 0, 0),
#     scale=0.02,
#     visualization=True,
#     fixed=True,
#     collision=True,
#     convexify=False
# ))

# scene.add_entity(gs.morphs.Mesh(
#     file="models/components/trashbin/Trashbin.glb",
#     pos=(0.55, 0, 0.05),
#     euler=(90, 0, 0),
#     scale=0.02,
#     visualization=True,
#     fixed=True,
#     collision=True,
#     convexify=False
# ))

scene.add_entity(gs.morphs.Mesh(
    file="models/components/yellowBin/yellowBin.obj",
    pos=(0.55, -0.15, 0.05),
    euler=(90, 0, 0),
    scale=0.2,
    visualization=True,
    fixed=True,
    collision=True,
    convexify=False,
))

# Build the scene
scene.build()

apple_pos = np.array(apple.get_pos().tolist())
print(apple_pos)

for _ in range(1000):
    scene.step()