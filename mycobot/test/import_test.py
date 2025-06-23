import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
       #  camera_pos=(1.2, 1.2, 0.8),
        camera_pos=(0, 1.2, 0.6),
        camera_lookat=(0, 0, 0),
        camera_fov=45,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)
scene.add_entity(gs.morphs.Plane())

mycobot = scene.add_entity(
    gs.morphs.URDF(
        file="/home/ubuntu1/models/mycobot_urdf_2022/mycobot_fixed_fully_connected.urdf",
        scale=0.001,
        pos=(0.0, 0.0, 0.0),
        euler=(0.0, 0.0, 0.0),
        fixed=True,
        collision=False,
    )
)

# 构建仿真
scene.build()

# 打印一下基本信息
print(f"Total DOF: {mycobot.n_dofs}")
print("Joint Names:", [j.name for j in mycobot.joints])
print("Link Names:", [l.name for l in mycobot.links])

for i in range(5000):
    scene.step()