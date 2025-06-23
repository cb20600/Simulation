import os
import genesis as gs
import numpy as np
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import sys

gs.init(backend=gs.gpu)

def create_scene():
    scene = gs.Scene(
        show_viewer = True,
        viewer_options = gs.options.ViewerOptions(
            res = (1280, 960),
            camera_pos =(1.5, 1.5, 1),
            camera_lookat = (0.0, 0.0, 0.2),
            camera_fov = 45,
            max_FPS = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True, # 显示原点坐标系
            world_frame_size = 1.0, # 坐标系长度(米)
            # ambient_light=(0.6, 0.6, 0.6),  # 默认只有 0.1，会让背光面变暗
            plane_reflection=False  # 可选：如果担心反光导致干扰
            # show_link_frame  = True, 
            # show_cameras = True, 
        ),
            show_FPS=False
    )


    table_height = 0.26
    scene.add_entity(gs.morphs.Plane())
    # scene.add_entity(gs.morphs.Plane(        
    #     pos=(-1.5, 0, 0),
    #     euler=(0, 90, 0),
    # ))
    # scene.add_entity(gs.morphs.Plane(        
    #     pos=(0, -2, 0),
    #     euler=(90, 0, 0),
    # ))

    xarm7 = scene.add_entity(gs.morphs.MJCF(
        file="models/ufactory_xarm7/xarm7.xml",
        collision=True,
        pos=(-0.3, 0.0, table_height)
        ),
        #vis_mode="collision"
        vis_mode="visual"
    )

    # 添加相机
    camera = scene.add_camera(
        res=(1280, 960),
        # res=(2560, 1440),
        pos=(0, 0.0, 1.0),    # 这里位置是临时的，后面会跟随机械臂末端
        lookat=(0.0, 0.0, 0.0),
        fov=60,
        GUI=True,
        # visuliza = True,
    )
    
    # 挂相机到机械臂末端
    link_name = "xarm_gripper_base_link"  # 末端夹爪基座 link
    gripper_link = xarm7.get_link(link_name)

    desk=scene.add_entity(gs.morphs.Mesh(
        file="models/components/dining_table_chair.glb",
        pos=(0, 0, 0),
        scale = 0.5,
        euler=(90, 0, 0),
        fixed=True,
        collision=True,
        convexify=False,
    ))

    fruit_x = 0.3
    fruit_y = 0.0
    fruit_z = 0.4

    banana_x = fruit_x + 0.05
    banana_y = fruit_y
    banana_z = fruit_z - 0.01

    carrot_x = fruit_x
    carrot_y = fruit_y - 0.05
    carrot_z = fruit_z - 0.01

    lemon_x = fruit_x-0.01
    lemon_y = fruit_y-0.06
    lemon_z = fruit_z + 0.01

    lime_x = fruit_x-0.05
    lime_y = fruit_y
    lime_z = fruit_z -0.01

    strawberry_x = fruit_x
    strawberry_y = fruit_y + 0.05
    strawberry_z = fruit_z - 0.01

    tomato_x = fruit_x
    tomato_y = fruit_y -0.05
    tomato_z = fruit_z

    banana = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/banana/banana.glb",
        pos = (banana_x,  banana_y, banana_z),
        euler=(0, 0, 60),
        scale=0.12,
        fixed=True,
        visualization=True,
        collision=True,
    ))

    carrot = scene.add_entity(
        gs.morphs.Mesh(
        file="models/components/fruits/carrot/carrot.glb",
        pos=(carrot_x, carrot_y, carrot_z),
        euler=(00, 90, -30),
        scale=0.006,
        fixed=True,
        visualization=True,
        collision=True,
    ))

    lemon = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/lemon/lemon.obj",
        pos = (lemon_x, lemon_y, lemon_z),
        euler=(90, 0, 60),
        scale=0.025,
        fixed=True,
        visualization=True,
        collision=True,
    ))    
    lime = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/lime/10187_Lime.obj",
        pos = (lime_x, lime_y, lime_z),
        euler=(90, 0, 60),
        scale=0.008,
        fixed=True,
        visualization=True,
        collision=True,
    ))    

    strawberry = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/strawberry/strawberry.glb",
        pos = (strawberry_x, strawberry_y, strawberry_z),
        euler=(0, 0, 60),
        scale=0.005,
        fixed=True,
        visualization=True,
        collision=True,
    ))   

    tomato = scene.add_entity(gs.morphs.Mesh(
        file="models/components/fruits/tomato/Tomato.obj",
        pos = (tomato_x, tomato_y, tomato_z),
        euler=(90, 0, 60),
        scale=0.025,
        fixed=True,
        visualization=True,
        collision=True,
    ))    

    # yellowBin =  scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="models/components/trashbin/Trashbin.glb",
    #         pos=(0.55, 0.1, 0.05),
    #         euler=(90, 0, 0),
    #         scale=0.02,
    #         visualization=True,
    #         fixed=True,
    #         collision=False,
    #         convexify=False
    #     ))

    # redBin =  scene.add_entity(
    #     gs.morphs.Mesh(
    #         file="models/components/trashbin/Trashbin.glb",
    #         pos=(0.55, -0.1, 0.05),
    #         euler=(90, 0, 0),
    #         scale=0.02,
    #         visualization=True,
    #         fixed=True,
    #         collision=False,
    #         convexify=False
    #     ))

    scene.build()

    T = np.array([
    [0,  -1,  0,  -0.08],
    [1,  0,  0,  0],
    [0,  0, -1,  0.02],
    [0,  0,  0,  1]
    ])
    # T = np.array([
    # [1,  0,  0,  -0.08],
    # [0, -1,  0,  0],
    # [0,  0, -1,  0],
    # [0,  0,  0,  1]
    # ])
    camera.attach(rigid_link=gripper_link, offset_T = T)
    
    fruit_entities = {
        "banana": banana,
        # "carrot": carrot,
        # "lemon": lemon,
        # "lime": lime,
        # "strawberry":strawberry,
        # "tomato": tomato
    }

    bins = {
        # "redBin": redBin,
        # "yellowBin": yellowBin,
    }


    return scene, xarm7, fruit_entities, bins, camera

if __name__ == "__main__":
    scene, xarm7, fruits, bins, camera = create_scene()

    for i in range(5000):
        scene.step()
        camera.render(
            rgb = True,
            # depth = True,
        )
    print("✅ 仿真完成")